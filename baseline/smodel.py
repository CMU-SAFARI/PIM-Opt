import os
import time
from copy import deepcopy
from dataclasses import dataclass
import queue

import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist

from sklearn.metrics import roc_auc_score

import pandas as pd

import numpy as np

from colored_logs import *
from model import *
from dataset import *


class SparseBinaryLinearModel:
  def __init__(self, dim=1000000, model_type="lr"):
    self.dim = dim
    self.weights    = torch.zeros(dim, dtype=torch.float32, requires_grad=False)
    self.bias       = 0.0
    self.grad       = torch.zeros(dim, dtype=torch.float32, requires_grad=False)
    self.bias_grad  = 0.0

    if model_type == "lr":
      self.loss = nn.BCEWithLogitsLoss()
      self.loss_no_reduce = nn.BCEWithLogitsLoss(reduction='none')
      self.backward = self.backward_lr
    else:
      self.loss = HingeLoss(margin=0.5, reduction="mean")
      self.loss_no_reduce = HingeLoss(margin=0.5, reduction='none')
      self.backward = self.backward_svm

  def zero_grad(self):
    self.grad.zero_()
    self.bias_grad = 0

  def forward(self, X):
    y_hat = torch.zeros(len(X), dtype=torch.float32, requires_grad=False)
    for i, x in enumerate(X):
      y_hat[i] = torch.sum(torch.index_select(self.weights, 0, x)) + self.bias
    return y_hat
  
  def backward_lr(self, y_hat, y, X):
    grad_update = torch.sigmoid(y_hat) - y
    grad_update = torch.where(y == 1.0, 15.0 * grad_update, 0.5 * grad_update)
    for i, x in enumerate(X):
      self.grad[x] += grad_update[i]
    self.grad /= len(X)
    self.bias_grad = torch.mean(grad_update)


  def backward_svm(self, y_hat, y, X):
    loss = 0.5 - y_hat * y
    grad_update = torch.where(loss >= 0, -y, 0)
    grad_update = torch.where(y == 1.0, 15.0 * grad_update, 0.5 * grad_update)
    for i, x in enumerate(X):
      self.grad[x] += grad_update[i]
    self.grad /= len(X)
    self.bias_grad = torch.mean(grad_update)


  def step_l2(self, lr, wd):
    self.grad *= -lr
    self.weights = self.weights + self.grad - lr * wd * self.weights
    self.bias    = self.bias - lr * self.bias_grad


  def step_l1(self, lr, wd):
    self.grad *= -lr
    l1_reg = torch.sgn(self.weights) * wd
    self.weights = self.weights + self.grad - lr * l1_reg
    self.bias    = self.bias - lr * self.bias_grad


class SparseBinaryLinearModelADMM:
  def __init__(self, dim=1000000, model_type="lr"):
    self.dim = dim
    self.weights    = torch.zeros(dim, dtype=torch.float32, requires_grad=False)
    self.bias       = torch.zeros(1, dtype=torch.float32, requires_grad=False)
    self.grad       = torch.zeros(dim, dtype=torch.float32, requires_grad=False)
    self.bias_grad  = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    if model_type == "lr":
      self.loss = nn.BCEWithLogitsLoss()
      self.loss_no_reduce = nn.BCEWithLogitsLoss(reduction='none')
      self.backward = self.backward_lr
    else:
      self.loss = HingeLoss(margin=0.5, reduction="mean")
      self.loss_no_reduce = HingeLoss(margin=0.5, reduction='none')
      self.backward = self.backward_svm

  def zero_grad(self):
    self.grad.zero_()
    self.bias_grad.zero_()

  def forward(self, X):
    y_hat = torch.zeros(len(X), dtype=torch.float32, requires_grad=False)
    for i, x in enumerate(X):
      y_hat[i] = torch.sum(torch.index_select(self.weights, 0, x)) + self.bias
    return y_hat
  
  def backward_lr(self, y_hat, y, X):
    grad_update = torch.sigmoid(y_hat) - y
    grad_update = torch.where(y == 1.0, 15.0 * grad_update, 0.5 * grad_update)
    for i, x in enumerate(X):
      self.grad[x] += grad_update[i]
    self.grad /= len(X)
    self.bias_grad = torch.mean(grad_update)


  def backward_svm(self, y_hat, y, X):
    loss = 0.5 - y_hat * y
    grad_update = torch.where(loss >= 0, -y, 0)
    grad_update = torch.where(y == 1.0, 15.0 * grad_update, 0.5 * grad_update)
    for i, x in enumerate(X):
      self.grad[x] += grad_update[i]
    self.grad /= len(X)
    self.bias_grad = torch.mean(grad_update)


  def step_l2(self, lr, wd):
    self.grad *= -lr
    self.weights = self.weights + self.grad - lr * wd * self.weights
    self.bias    = self.bias - lr * self.bias_grad


  def step_l1(self, lr, wd):
    self.grad *= -lr
    l1_reg = torch.sgn(self.weights) * wd
    self.weights = self.weights + self.grad - lr * l1_reg
    self.bias    = self.bias - lr * self.bias_grad

def train_CPU_sparse(rank, dataset, args):
  if rank == 0:
    logger = setup_logger()

  # Initialize DDP
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '19244'
  dist.init_process_group("gloo", rank=rank, world_size=args.num_procs)

  dataset.load_rank_data(args.num_procs, rank, args.batch_size, args.dataset_size)
  # dataset = CriteoDense("test_data.pt", logger=None)
  # Model
  if args.optim == "admm":
    model = SparseBinaryLinearModelADMM(dim=dataset.num_features, model_type=args.model_type)
  elif args.optim == "sgd":
    model = SparseBinaryLinearModel(dim=dataset.num_features, model_type=args.model_type)

  torch.set_num_threads(1)

  # Optimizer
  lr = pow(2, -args.lr)
  weight_decay = pow(2, -args.wd)

  # Initialize ADMM stuff. Rank 0 acts as the host here
  if args.optim == "admm":
    alpha = pow(2, -args.alpha)
    if rank == 0:
      w_local_list = [torch.zeros_like(model.weights)] * args.num_procs
      u_local_list = [torch.zeros_like(model.weights)] * args.num_procs
      w_global = torch.zeros_like(model.weights)
      u_global = torch.zeros_like(model.weights)
      z_global = torch.zeros_like(model.weights)
      u_z_local_list = [torch.zeros_like(model.weights)] * args.num_procs

      w_bias_local_list = [torch.zeros(1)] * args.num_procs
      u_bias_local_list = [torch.zeros(1)] * args.num_procs
      w_bias_global = torch.zeros(1)
      u_bias_global = torch.zeros(1)
      z_bias_global = torch.zeros(1)
      u_z_bias_local_list = [torch.zeros(1)] * args.num_procs
    else:
      w_local_list = None
      u_local_list = None
      u_z_local_list = None
      w_bias_local_list = None
      u_bias_local_list = None
      u_z_bias_local_list = None


    u_z_local = torch.zeros_like(model.weights)
    u_z_bias_local = torch.zeros(1)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

  # We keep a list of (averaged) model weights on rank 0 for every global epoch
  epoch_data = [] if rank == 0 else None

  # Training starts
  if rank == 0:
    logger.info(f"Started training ({args}) ...")
  total_training_start = time.time()

  # SGD
  if args.optim in ["sgd", "adam"]:
    # Turn off grad sync since we are averaging the model
    if args.dist_type == "ma":
      with torch.no_grad():
        for global_epoch_id in range(args.num_global_epochs):
          global_epoch_start = time.time()
          # Local epoch training (no sync)
          for local_epoch_id in range(args.num_local_epochs):
            for batch_id, (X, y) in enumerate(dataloader):
              # loss = criterion(model(X), y.view(-1, 1)) + weight_decay * torch.norm(model.module.linear.weight.data, p=1)
              y = y.view(-1, 1)
              y_hat = model.forward(X).view(-1, 1)

              # Do not need to compute loss during training
              # loss = model.loss_no_reduce(y_hat, y)
              # loss = torch.where(y == 1.0, 15.0 * loss, 0.5 * loss)
              # loss = torch.mean(loss)

              model.zero_grad()
              model.backward(y_hat, y, X)
              model.step_l2(lr, weight_decay)

          global_epoch_training_end = time.time()
          global_epoch_training_time = global_epoch_training_end - global_epoch_start

          # Sync and average the model weights at the end of each global epoch
          weight_reduce = dist.all_reduce(model.weights, op=dist.ReduceOp.SUM, async_op=True)
          bias_reduce = dist.all_reduce(model.bias, op=dist.ReduceOp.SUM, async_op=True)
          weight_reduce.wait()
          bias_reduce.wait()
          model.weights /= args.num_procs
          model.bias /= args.num_procs

          global_epoch_end = time.time()
          global_epoch_sync_time = global_epoch_end - global_epoch_training_end
          global_epoch_time = global_epoch_end - global_epoch_start


          # y_hat_all = []
          # y_all = []

          # train_loss = 0.0
          # for batch_id, (X, y) in enumerate(dataloader):
          #   y = y.view(-1, 1)
          #   y_hat = model.forward(X).view(-1, 1)
          #   loss = model.loss_no_reduce(y_hat, y)
          #   loss = torch.where(y == 1.0, 15.0 * loss, 0.5 * loss)
          #   train_loss += torch.sum(loss)

          #   y_hat_all.append(torch.sigmoid(y_hat.view(1, -1)).flatten())
          #   y_all.append(y.view(1, -1).flatten())

          # dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)

          # df = pd.DataFrame(model.weights.numpy())
          # df.to_csv(f"weights_{global_epoch_id}.csv",index=False)

          # if rank == 0:
          #   logger.info(f"Bias {model.bias}")


          # y_list = torch.empty(args.num_procs, dtype=torch.float32)
          # y_hat_list = torch.empty(args.num_procs, dtype=torch.float32)

          # y_all = torch.cat(y_all)
          # y_hat_all = torch.cat(y_hat_all)

          # # if rank == 0:
          # #   torch.gather(y_all, y_list)
          # #   torch.gather(y_hat_all, y_hat_list)
          # # else:
          # #   torch.gather(y_all, None)
          # #   torch.gather(y_hat_all, None)

          # # y_all = torch.cat(y_all)
          # # y_hat_all = torch.cat(y_hat_all)
          # y_zero = torch.zeros_like(y_hat_all)

          # # y_hat_alt = log_reg_compute_probs(args.dataset_size, )

          # # if rank == 0:
          # #   logger.info(f"y_true set idx = {torch.where(y_all == 1.0)}")

          # score = roc_auc_score(y_all.numpy(), y_hat_all.numpy())
          # base_score = roc_auc_score(y_all.numpy(), y_zero.numpy())

          # Save the model weights and statistics of this global epoch
          if rank == 0:
            logger.info(f"Global epoch {global_epoch_id} finished.")
            # logger.info(f"Train Loss = {train_loss / (len(dataset) * args.num_procs)}")
            # logger.info(f"ROC AUC Score = {score}")
            # logger.info(f"Baseline ROC AUC Score = {base_score}")

            epoch_data.append({
              "global_epoch_id" : global_epoch_id,
              "model_weight" : deepcopy(model.weights),
              "model_bias" : deepcopy(model.bias),
              "training_time" : global_epoch_training_time,
              "sync_time" : global_epoch_sync_time,
              "total_time" : global_epoch_time,
            })

    elif args.dist_type == "ma1":
      with torch.no_grad():
        for global_epoch_id in range(args.num_global_epochs):
          # Local epoch training (no sync)
          global_epoch_start = time.time()

          # Local epoch training (no sync)
          global_epoch_training_time = 0.0
          global_epoch_comm_time = 0.0

          for batch_id, (X, y) in enumerate(dataloader):
            batch_training_start = time.time()

            y = y.view(-1, 1)
            y_hat = model.forward(X).view(-1, 1)

            model.zero_grad()
            model.backward(y_hat, y, X)
            model.step_l2(lr, weight_decay)

            batch_training_end = time.time()
            global_epoch_training_time += batch_training_end - batch_training_start

            # Sync and average the model weights at the end of each global epoch
            weight_reduce = dist.all_reduce(model.weights, op=dist.ReduceOp.SUM, async_op=True)
            bias_reduce = dist.all_reduce(model.bias, op=dist.ReduceOp.SUM, async_op=True)
            weight_reduce.wait()
            bias_reduce.wait()
            model.weights /= args.num_procs
            model.bias /= args.num_procs

            batch_comm_end = time.time()
            global_epoch_comm_time += batch_comm_end - batch_training_end

          global_epoch_end = time.time()
          global_epoch_time = global_epoch_end - global_epoch_start

          if rank == 0:
            logger.info(f"Global epoch {global_epoch_id} finished.")
            # logger.info(f"Train Loss = {train_loss / (len(dataset) * args.num_procs)}")
            # logger.info(f"ROC AUC Score = {score}")
            # logger.info(f"Baseline ROC AUC Score = {base_score}")

            epoch_data.append({
              "global_epoch_id" : global_epoch_id,
              "model_weight" : deepcopy(model.weights),
              "model_bias" : deepcopy(model.bias),
              "training_time" : global_epoch_training_time,
              "sync_time" : global_epoch_comm_time,
              "total_time" : global_epoch_time,
            })

    elif args.dist_type == "ga":
      with torch.no_grad():
        for global_epoch_id in range(args.num_global_epochs):
          global_epoch_start = time.time()
          global_epoch_sync_time = 0.0
          # Local epoch training (no sync)
          # for local_epoch_id in range(args.num_local_epochs):
          for batch_id, (X, y) in enumerate(dataloader):
            # loss = criterion(model(X), y.view(-1, 1)) + weight_decay * torch.norm(model.module.linear.weight.data, p=1)
            y_hat = model.forward(X)
            loss = model.loss(y_hat, y)
            model.zero_grad()
            model.backward(y_hat, y, X)

            # Average the gradients from all workers
            batch_sync_start = time.time()
            dist.all_reduce(model.grad, op=dist.ReduceOp.SUM)
            global_epoch_sync_time += time.time() - batch_sync_start
            model.grad /= args.num_procs

            model.step_l2(lr, weight_decay)
            
          global_epoch_training_end = time.time()
          global_epoch_training_time = global_epoch_training_end - global_epoch_start

          # Save the model weights and statistics of this global epoch
          if rank == 0:
            logger.info(f"Global epoch {global_epoch_id} finished.")
            epoch_data.append({
              "global_epoch_id" : global_epoch_id,
              "model_weight" : deepcopy(model.weights),
              "model_bias" : deepcopy(model.bias),
              "training_time" : global_epoch_training_time,
              "sync_time" : global_epoch_sync_time,
              "total_time" : -1,
            })

  # ADMM
  elif args.optim == "admm":
    with torch.no_grad():
    # Turn off grad sync since we are averaging the model
    # with model.no_sync():
      for global_epoch_id in range(args.num_global_epochs):
        global_epoch_start = time.time()

        # Local epoch training (no sync)
        for local_epoch_id in range(args.num_local_epochs):
          for batch_id, (X, y) in enumerate(dataloader):
            y = y.view(-1, 1)
            y_hat = model.forward(X).view(-1, 1)
            model.zero_grad()
            model.backward(y_hat, y, X)
            if dataset.model_type == "svm":
              model.step_l2(lr, weight_decay)
            elif dataset.model_type == "lr":
              model.step_l1(lr, weight_decay)

        global_epoch_training_end = time.time()
        global_epoch_training_time = global_epoch_training_end - global_epoch_start

        # Communicate w_local with host
        dist.gather(model.weights, gather_list=w_local_list, dst=0)
        dist.gather(model.bias,   gather_list=w_bias_local_list, dst=0)

        # Update w, u, z, and u_z
        if rank == 0:
          w_global = torch.stack(w_local_list).mean(dim=0)
          u_global = torch.stack(u_local_list).mean(dim=0)
          z_global = update_z(w_global, u_global, weight_decay, args.num_procs, alpha)

          w_bias_global = torch.stack(w_bias_local_list).mean(dim=0)
          u_bias_global = torch.stack(u_bias_local_list).mean(dim=0)
          z_bias_global = update_z(w_bias_global, u_bias_global, weight_decay, args.num_procs, alpha)

          for i in range(args.num_procs):
            u_local_list[i] += w_local_list[i] - z_global
            u_z_local_list[i] = u_local_list[i] - z_global

            u_bias_local_list[i] += w_bias_local_list[i] - z_bias_global
            u_z_bias_local_list[i] = u_bias_local_list[i] - z_bias_global

        # Communicate u zloca
        dist.scatter(u_z_local, scatter_list=u_z_local_list, src=0)
        dist.scatter(u_z_bias_local, scatter_list=u_z_bias_local_list, src=0)

        global_epoch_end = time.time()
        global_epoch_sync_time = global_epoch_end - global_epoch_training_end
        global_epoch_time = global_epoch_end - global_epoch_start

        # Save the model weights and statistics of this global epoch
        if rank == 0:
          logger.info(f"Global epoch {global_epoch_id} finished.")
          epoch_data.append({
            "global_epoch_id" : global_epoch_id,
            "model_weight" : deepcopy(model.weights),
            "model_bias" : deepcopy(model.bias),
            "z_global" : deepcopy(z_global.data),
            "training_time" : global_epoch_training_time,
            "sync_time" : global_epoch_sync_time,
            "total_time" : global_epoch_time,
          })
  else:
    raise RuntimeError(f"Unrecognized optimizer {args.optim}!")
  
  # Save epoch data to a .pt file to be evaluated separately
  total_training_time = time.time() - total_training_start
  if rank == 0:
    logger.info(f"Finished training.")
    torch.save({
        "args" : args,
        "epoch_data" : epoch_data,
        "total_training_time": total_training_time,
      },
      f"{args.device}_{args.dataset}_{args.model_type}_{args.optim}_{args.dist_type}_{args.num_global_epochs}_{args.num_local_epochs}_{args.batch_size}_{args.lr}_{args.wd}_{args.alpha}_{args.seed}_{args.num_procs}.pt"
    )

  # Cleanup
  dist.destroy_process_group()


class SparseBinaryLinearModelGPU:
  def __init__(self, dim=1000000, model_type="lr", device="cuda:0"):
    self.device = device
    self.dim = dim
    self.weights    = torch.zeros(dim, dtype=torch.float32, requires_grad=False, device=device)
    self.bias       = torch.zeros(1,   dtype=torch.float32, requires_grad=False, device=device)
    self.grad       = torch.zeros(dim, dtype=torch.float32, requires_grad=False, device=device)
    self.bias_grad  = torch.zeros(1,   dtype=torch.float32, requires_grad=False, device=device)

    if model_type == "lr":
      self.loss = nn.BCEWithLogitsLoss()
      self.loss_no_reduce = nn.BCEWithLogitsLoss(reduction='none')
      self.backward = self.backward_lr
    else:
      self.loss = HingeLoss(margin=0.5, reduction="mean")
      self.loss_no_reduce = HingeLoss(margin=0.5, reduction='none')
      self.backward = self.backward_svm

  def zero_grad(self):
    self.grad.zero_()
    self.bias_grad.zero_()

  def forward(self, X):
    y_hat = torch.zeros(len(X), dtype=torch.float32, requires_grad=False, device=self.device)
    for i, x in enumerate(X):
      y_hat[i] = torch.sum(torch.index_select(self.weights, 0, x)) + self.bias
    return y_hat
  
  def backward_lr(self, y_hat, y, X):
    grad_update = torch.sigmoid(y_hat) - y
    grad_update = torch.where(y == 1.0, 15.0 * grad_update, 0.5 * grad_update)
    for i, x in enumerate(X):
      self.grad[x] += grad_update[i]
    self.grad /= len(X)
    self.bias_grad = torch.mean(grad_update)


  def backward_svm(self, y_hat, y, X):
    loss = torch.clamp(0.5 - y_hat * y, min=0)
    grad_update = torch.where(loss >= 0, -y, 0)
    grad_update = torch.where(y == 1.0, 15.0 * grad_update, 0.5 * grad_update)
    for i, x in enumerate(X):
      self.grad[x] += grad_update[i]
    self.grad /= len(X)
    self.bias_grad = torch.mean(grad_update)


  def step(self, lr, wd):
    self.grad *= -lr
    self.weights = self.weights + self.grad - lr * wd * self.weights
    self.bias    = self.bias - lr * self.bias_grad


def train_GPU_sparse(dataset, args, logger = None):
  device = torch.device("cuda:0")
  # Enable TF32 to use Tensorcores for MatMul
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

  # Model
  model = SparseBinaryLinearModelGPU(dim=dataset.num_features, model_type=args.model_type)
  
  lr = pow(2, -args.lr)
  weight_decay = pow(2, -args.wd)

  dataloader_kwargs = {
  'batch_size': args.batch_size,
  'shuffle': False,   # We dont shuffle here as the train data is pre-shuffled.
  'num_workers': 8,
  'pin_memory': True,
  'pin_memory_device': "cuda:0"
  }
  dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

  # Minibatch prefetcher
  batch_queue = queue.Queue(maxsize=args.buffer_size)
  data_iter = iter(dataloader)
  for _ in range(batch_queue.maxsize):
    try:
      X, y = next(data_iter)
      X = X.to(device)
      y = y.view(-1, 1).to(device)
      batch_queue.put((X, y))
    except StopIteration:
      break

  # Save model for every epoch
  epoch_data = []
  batch_times = []
  # Training
  logger.info(f"Started training ({args}) ...")
  total_start_event = torch.cuda.Event(enable_timing=True)
  total_end_event = torch.cuda.Event(enable_timing=True)
  total_start_event.record()
  for g_epoch in range(args.num_global_epochs):
    epoch_start_event = torch.cuda.Event(enable_timing=True)
    epoch_end_event = torch.cuda.Event(enable_timing=True)
    epoch_start_event.record()
    for batch_id in range(len(dataloader)):
      # batch_start_event = torch.cuda.Event(enable_timing=True)
      # batch_end_event = torch.cuda.Event(enable_timing=True)
      # batch_start_event.record()
      X, y = batch_queue.get()
      y = y.view(-1, 1)
      y_hat = model.forward(X).view(-1, 1)

      model.zero_grad()
      model.backward(y_hat, y, X)
      model.step(lr, weight_decay)

      try:
        X, y = next(data_iter)
        X = X.to(device)
        y = y.view(-1, 1).to(device)
        batch_queue.put((X, y))
      except StopIteration:
        break

      # batch_end_event.record()
      # torch.cuda.synchronize()
      # batch_time = batch_start_event.elapsed_time(batch_end_event)
      # batch_times.append(batch_time)
      # logger.info(f"Batch {batch_id} took {batch_start_event.elapsed_time(batch_time)}.")
      if batch_id == 20:
        break

    epoch_end_event.record()
    torch.cuda.synchronize()
    epoch_time_ms = epoch_start_event.elapsed_time(epoch_end_event)
    epoch_data.append({
        "global_epoch_id": g_epoch,
        "model_weight": deepcopy(model.weights),
        "model_bias": deepcopy(model.bias),
        "total_time": epoch_time_ms,
        "batch_times": batch_times,
    })
  total_end_event.record()
  torch.cuda.synchronize()
  total_time_ms = total_start_event.elapsed_time(total_end_event)
  logger.info(f"Finished training.")
  torch.save({
      "args" : args,
      "epoch_data" : epoch_data,
      "total_training_time": total_time_ms,
    },
    f"{args.device}_{args.dataset}_{args.model_type}_{args.optim}_{args.num_global_epochs}_{args.batch_size}_{args.lr}_{args.wd}_{args.seed}.pt"
  )


def eval_CPU_sparse(rank, world_size, train_dataset, train_dataset_size, test_dataset, test_dataset_size, data_path, results_filename):
  df_lines = None
  logger = None
  if rank == 0:
    logger = setup_logger()
    df_lines = []

  # Initialize DDP
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '19244'
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

  train_dataset.load_rank_data(world_size, rank, 1024, train_dataset_size)
  test_dataset.load_rank_data(world_size, rank, 1024, test_dataset_size)
  dist.barrier()
  torch.set_num_threads(1)

  result_files = []
  for root, dirs, files in os.walk(data_path):
    for file in files:
      if file.endswith(".pt"):
        result_files.append(file)

  for file in result_files:
    if rank == 0:
      logger.info(f"Evaluating {file}...")


    data = torch.load(os.path.join(root, file), map_location=torch.device('cpu'))
    args = data["args"]

    model = SparseBinaryLinearModel(dim=train_dataset.num_features)
    
    # # Loss
    # criterion = None
    # if train_dataset.model_type == "lr":
    #   criterion = nn.BCEWithLogitsLoss(reduction="sum")
    # elif train_dataset.model_type == "svm":
    #   criterion = HingeLoss(margin=0.5, reduction="sum")
    # else:
    #   raise RuntimeError(f"Unrecognized model type {train_dataset.model_type}!")

    weight_decay = pow(2, -args.wd)

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1024)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

    # Evaluation starts
    if rank == 0:
      logger.info(f"Started evaluation ({args}) ...")

    with torch.no_grad():
      for global_epoch_id in range(args.num_global_epochs):
        train_loss = 0.0
        test_loss = 0.0

        # Load the model parameters at this global epoch
        model.weights = data["epoch_data"][global_epoch_id]["model_weight"]
        model.bias = data["epoch_data"][global_epoch_id]["model_bias"]

        train_loss = 0.0
        y_hat_train_list = []
        y_train_list = []
        for batch_id, (X, y) in enumerate(train_dataloader):
          y = y.view(-1, 1)
          y_hat = model.forward(X).view(-1, 1)
          loss = model.loss_no_reduce(y_hat, y)
          loss = torch.where(y == 1.0, 15.0 * loss, 0.5 * loss)
          train_loss += torch.sum(loss)

          y_hat_train_list.append(torch.sigmoid(y_hat.view(1, -1)).flatten())
          if args.model_type == "lr":
            y_train_list.append(y.view(1, -1).flatten())
          elif args.model_type == "svm":
            y_train_list.append(torch.where(y.view(1, -1).flatten() == -1.0, 0.0, 1.0))

        test_loss = 0.0
        y_hat_test_list = []
        y_test_list = []
        for batch_id, (X, y) in enumerate(test_dataloader):
          y = y.view(-1, 1)
          y_hat = model.forward(X).view(-1, 1)
          loss = model.loss_no_reduce(y_hat, y)
          loss = torch.where(y == 1.0, 15.0 * loss, 0.5 * loss)
          test_loss += torch.sum(loss)

          y_hat_test_list.append(torch.sigmoid(y_hat.view(1, -1)).flatten())
          if args.model_type == "lr":
            y_test_list.append(y.view(1, -1).flatten())
          elif args.model_type == "svm":
            y_test_list.append(torch.where(y.view(1, -1).flatten() == -1.0, 0.0, 1.0))
          
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)

        train_loss /= (len(train_dataset) * world_size)
        test_loss /= (len(test_dataset) * world_size)

        y_train = torch.cat(y_train_list)
        y_hat_train = torch.cat(y_hat_train_list)
        y_test = torch.cat(y_test_list)
        y_hat_test = torch.cat(y_hat_test_list)

        all_train_y_list = [torch.zeros_like(y_train, dtype=torch.float32) for _ in range(world_size)]
        all_train_y_hat_list = [torch.zeros_like(y_hat_train, dtype=torch.float32) for _ in range(world_size)]
        all_test_y_list = [torch.zeros_like(y_hat_test, dtype=torch.float32) for _ in range(world_size)]
        all_test_y_hat_list = [torch.zeros_like(y_hat_test, dtype=torch.float32) for _ in range(world_size)]


        if rank == 0:
          dist.gather(y_train, all_train_y_list)
          dist.gather(y_hat_train, all_train_y_hat_list)
          dist.gather(y_test, all_test_y_list)
          dist.gather(y_hat_test, all_test_y_hat_list)
        else:
          dist.gather(y_train, None)
          dist.gather(y_hat_train, None)
          dist.gather(y_test, None)
          dist.gather(y_hat_test, None)


        if rank == 0:
          all_train_y = torch.cat(all_train_y_list)
          all_train_y_hat = torch.cat(all_train_y_hat_list)
          all_test_y = torch.cat(all_test_y_list)
          all_test_y_hat = torch.cat(all_test_y_hat_list)

          train_ra_score = roc_auc_score(all_train_y.numpy(), all_train_y_hat.numpy())
          test_ra_score = roc_auc_score(all_test_y.numpy(), all_test_y_hat.numpy()) 
        

        if rank == 0:
          model_norm = torch.linalg.norm(model.weights)
          l2_reg = 0.5 * model_norm * model_norm * weight_decay

          # training_time = data["epoch_data"][global_epoch_id]["training_time"]
          # sync_time = data["epoch_data"][global_epoch_id]["sync_time"]
          total_time = data["epoch_data"][global_epoch_id]["total_time"]
          if args.dist_type == "ga":
            total_time = data["epoch_data"][global_epoch_id]["training_time"]
          # logger.info(f"Epoch {global_epoch_id}: Train loss {train_loss}, acc {train_accuracy}%. Test loss {test_loss}, acc {test_accuracy}%. L2_reg {l2_reg}.")
          # logger.info(f"Epoch {global_epoch_id}: Total time {total_time}s (training {training_time}s, sync {sync_time}s)")          
          logger.info(f"Epoch {global_epoch_id}: Train loss {train_loss}, acc {train_ra_score}%. Test loss {test_loss}, acc {test_ra_score}%. L2_reg {l2_reg}.")
          logger.info(f"Epoch {global_epoch_id}: Total time {total_time}s")          

          if args.optim == "sgd":
            args.optim = "mbsgd"
            args.alpha = np.nan

          df_lines.append(
            [
              args.device.upper(), args.dataset, args.model_type, args.optim, "float32",
              args.num_global_epochs, args.num_local_epochs, args.batch_size, args.lr, args.wd, args.alpha, args.dist_type,
              args.seed,
              args.num_procs,

              global_epoch_id,
              train_loss.item(), train_ra_score, test_loss.item(), test_ra_score,
              l2_reg.item(),
              total_time,
            ]
          )

  # Save results to a csv
  if rank == 0:
    df = pd.DataFrame(df_lines)
    df.columns = [
      "architecture", "dataset", "model_type", "algorithm", "type_precision",
      "num_global_epochs", "num_local_epochs", "batch_size", "lr", "reg_term", "reg_term_alpha", "dist_type",
      "seed",
      "num_procs",

      "g_epoch_id",
      "train_loss", "train_accuracy", "test_loss", "test_accuracy", 
      "l2_reg", 
      "epoch_time",
    ]
    
    df.to_csv(f"{results_filename}", index=False)

  # Cleanup
  dist.destroy_process_group()
