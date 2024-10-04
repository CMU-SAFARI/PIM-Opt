import os
import time
from copy import deepcopy
from dataclasses import dataclass
import queue

import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist

import pandas as pd

import numpy as np

from colored_logs import *
from model import *
from dataset import *


class LinearModel(nn.Module):
  """
  Linear model used for logistic regression (sigmoid in BCEwithLogisticsLoss) and SVM. 
  All weights are initialized to 0. Bias is disabled to be the same as the UPMEM implementation.
  """
  def __init__(self, input_dim):
    super(LinearModel, self).__init__()
    self.linear = nn.Linear(input_dim, 1, bias=True)
    self.linear.weight.data.fill_(0)
    self.linear.bias.data.fill_(0)

  def forward(self, x):
    out = self.linear(x)
    return out
  

class HingeLoss(nn.Module):
  """
  Hinge loss function used in SVM
  """
  def __init__(self, margin, reduction="mean"):
    super(HingeLoss, self).__init__()
    self.margin = margin
    if reduction == "mean":
      self.reduction = torch.mean
    elif reduction == "sum":
      self.reduction = torch.sum
    elif reduction == "none":
      self.reduction = lambda x: x
    else:
      raise RuntimeError(f"Unrecognized reduction method {reduction}!")
    
  def forward(self, output, target):
    loss = torch.clamp(self.margin - output * target, min=0)
    loss = self.reduction(loss)
    return loss


def update_z(w, u, rho, n, lam_0):
  """
  Helper function to update z in ADMM (From https://github.com/DS3Lab/LambdaML)
  """
  z_new = w + u
  z_tem = abs(z_new) - lam_0 / float(n * rho)
  z_new = torch.sign(z_new) * z_tem * (z_tem > 0)
  return z_new


@dataclass
class TrainArgs:
  device: str = "UNKNOWN"       # device type ("cpu" or "gpu")
  dataset: str = "UNKNOWN"      # dataset ("yfcc" or "criteo")
  model_type: str = "UNKNOWN"   # model type ("lr" or "svm")
  optim: str = "UNKNOWN"        # optimizer type ("sgd", "admm", or "adam")

  dist_type: str = "UNKNOWN"    # Distributed training type ("ma", or "ga")

  num_global_epochs: int = 10   # number of global epochs for training (default: 10)
  num_local_epochs: int = 2     # number of local epochsper global epoch (default: 2)
  batch_size: int = 256         # size of each minibatch
  lr: int = 8                   # learning rate in the form of 2^-N
  # l1_wd: int = 8              # L1 weight decay in the form of 2^-N
  wd: int = 8                   # L2 weight decay in the form of 2^-N
  alpha: int = 12               # ADMM's alpha in the form of 2^-N

  seed: int = 1                 # random seed

  num_procs: int = 32           # number of CPU processes
  buffer_size: int = 10         # The minibatch buffer size for GPU


def train_CPU(rank, dataset, args):
  if rank == 0:
    logger = setup_logger()

  # Initialize DDP
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '19244'
  dist.init_process_group("gloo", rank=rank, world_size=args.num_procs)

  torch.set_num_threads(1)

  # Model
  model = LinearModel(dataset.num_features)
  model = torch.nn.parallel.DistributedDataParallel(model)
  model.train()

  # Loss
  criterion = None
  if dataset.model_type == "lr":
    criterion = nn.BCEWithLogitsLoss()
  elif dataset.model_type == "svm":
    criterion = HingeLoss(margin=0.5)
  else:
    raise RuntimeError(f"Unrecognized model type {dataset.model_type}!")

  # Optimizer
  lr = pow(2, -args.lr)
  weight_decay = pow(2, -args.wd)
  if args.optim == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
  elif args.optim == "admm":
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
  elif args.optim == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr)

  # Initialize ADMM stuff. Rank 0 acts as the host here
  if args.optim == "admm":
    alpha = pow(2, -args.alpha)
    if rank == 0:
      w_local_list = [torch.zeros_like(model.module.linear.weight.data)] * args.num_procs
      u_local_list = [torch.zeros_like(model.module.linear.weight.data)] * args.num_procs
      w_global = torch.zeros_like(model.module.linear.weight.data)
      u_global = torch.zeros_like(model.module.linear.weight.data)
      z_global = torch.zeros_like(model.module.linear.weight.data)
      u_z_local_list = [torch.zeros_like(model.module.linear.weight.data)] * args.num_procs

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


    u_z_local = torch.zeros_like(model.module.linear.weight.data)
    u_z_bias_local = torch.zeros(1)

  sampler = torch.utils.data.distributed.DistributedSampler(dataset, args.num_procs, rank, shuffle=False, seed=args.seed)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

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
      with model.no_sync():
        for global_epoch_id in range(args.num_global_epochs):
          global_epoch_start = time.time()

          # Local epoch training (no sync)
          for local_epoch_id in range(args.num_local_epochs):
            for batch_id, (X, y) in enumerate(dataloader):
              # loss = criterion(model(X), y.view(-1, 1)) + weight_decay * torch.norm(model.module.linear.weight.data, p=1)
              loss = criterion(model(X), y.view(-1, 1))
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
            
          global_epoch_training_end = time.time()
          global_epoch_training_time = global_epoch_training_end - global_epoch_start

          # Sync and average the model weights at the end of each global epoch
          with torch.no_grad():
            dist.all_reduce(model.module.linear.weight.data, op=dist.ReduceOp.SUM)
            dist.all_reduce(model.module.linear.bias.data, op=dist.ReduceOp.SUM)
            model.module.linear.weight.data /= args.num_procs
            model.module.linear.bias.data /= args.num_procs

          global_epoch_end = time.time()
          global_epoch_sync_time = global_epoch_end - global_epoch_training_end
          global_epoch_time = global_epoch_end - global_epoch_start

          # Save the model weights and statistics of this global epoch
          if rank == 0:
            logger.info(f"Global epoch {global_epoch_id} finished.")
            epoch_data.append({
              "global_epoch_id" : global_epoch_id,
              "model_weight" : deepcopy(model.module.linear.weight.data),
              "model_bias" : deepcopy(model.module.linear.bias.data),
              "training_time" : global_epoch_training_time,
              "sync_time" : global_epoch_sync_time,
              "total_time" : global_epoch_time,
            })
    elif args.dist_type == "ma1":
      # Calculate the synchronization time
      with model.no_sync():
        for global_epoch_id in range(args.num_global_epochs):
          global_epoch_start = time.time()

          # Local epoch training (no sync)
          global_epoch_training_time = 0.0
          global_epoch_comm_time = 0.0
          for batch_id, (X, y) in enumerate(dataloader):
            # loss = criterion(model(X), y.view(-1, 1)) + weight_decay * torch.norm(model.module.linear.weight.data, p=1)
            batch_training_start = time.time()

            loss = criterion(model(X), y.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            batch_training_end = time.time()
            global_epoch_training_time += batch_training_end - batch_training_start

            # Sync and average the model weights at the end of each global epoch
            with torch.no_grad():
              dist.all_reduce(model.module.linear.weight.data, op=dist.ReduceOp.SUM)
              dist.all_reduce(model.module.linear.bias.data, op=dist.ReduceOp.SUM)
              model.module.linear.weight.data /= args.num_procs
              model.module.linear.bias.data /= args.num_procs
            batch_comm_end = time.time()
            global_epoch_comm_time += batch_comm_end - batch_training_end

          global_epoch_end = time.time()
          global_epoch_time = global_epoch_end - global_epoch_start

          # Save the model weights and statistics of this global epoch
          if rank == 0:
            logger.info(f"Global epoch {global_epoch_id} finished.")
            epoch_data.append({
              "global_epoch_id" : global_epoch_id,
              "model_weight" : deepcopy(model.module.linear.weight.data),
              "model_bias" : deepcopy(model.module.linear.bias.data),
              "training_time" : global_epoch_training_time,
              "sync_time" : global_epoch_comm_time,
              "total_time" : global_epoch_time,
            })

    elif args.dist_type == "ga":
      for global_epoch_id in range(args.num_global_epochs):
        global_epoch_start = time.time()
        loss_acc = 0
        for local_epoch_id in range(args.num_local_epochs):
          for batch_id, (X, y) in enumerate(dataloader):
            # loss = criterion(model(X), y.view(-1, 1)) + weight_decay * torch.norm(model.module.linear.weight.data, p=1)
            loss = criterion(model(X), y.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_acc += loss

        global_epoch_training_end = time.time()
        global_epoch_training_time = global_epoch_training_end - global_epoch_start

        if rank == 0:
          loss_acc /= (len(dataset) / args.num_procs)
          logger.info(f"Global epoch {global_epoch_id}: Loss = {loss_acc}")

        # Save the model weights and statistics of this global epoch
        if rank == 0:
          logger.info(f"Global epoch {global_epoch_id} finished.")
          epoch_data.append({
            "global_epoch_id" : global_epoch_id,
            "model_weight" : deepcopy(model.module.linear.weight.data),
            "model_bias" : deepcopy(model.module.linear.bias.data),
            "training_time" : global_epoch_training_time,
            "sync_time" : -1.0,
            "total_time" : global_epoch_training_time,
          })

  # ADMM
  elif args.optim == "admm":
    # Turn off grad sync since we are averaging the model
    with model.no_sync():
      for global_epoch_id in range(args.num_global_epochs):
        global_epoch_start = time.time()

        # Local epoch training (no sync)
        for local_epoch_id in range(args.num_local_epochs):
          for batch_id, (X, y) in enumerate(dataloader):
            loss = criterion(model(X), y.view(-1, 1))
            if dataset.model_type == "lr":
              loss += weight_decay * torch.norm(model.module.linear.weight + u_z_local, p=1)
            elif dataset.model_type == "svm":
              loss += weight_decay / 2.0 * torch.norm(model.module.linear.weight + u_z_local, p=2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        global_epoch_training_end = time.time()
        global_epoch_training_time = global_epoch_training_end - global_epoch_start

        # Communicate w_local with host
        dist.gather(model.module.linear.weight.data, gather_list=w_local_list, dst=0)
        dist.gather(model.module.linear.bias.data,   gather_list=w_bias_local_list, dst=0)

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
            "model_weight" : deepcopy(model.module.linear.weight.data),
            "model_bias" : deepcopy(model.module.linear.bias.data),
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


def train_GPU(dataset, args, logger = None):
  device = torch.device("cuda:0")
  # Enable TF32 to use Tensorcores for MatMul
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

  # Model
  model = LinearModel(dataset.num_features).to(device)
  model.train()
  
  # Loss
  criterion = None
  if dataset.model_type == "lr":
    criterion = nn.BCEWithLogitsLoss()
  elif dataset.model_type == "svm":
    criterion = HingeLoss(margin=0.5)
  else:
    raise RuntimeError(f"Unrecognized model type {dataset.model_type}!")

  # Optimizer
  lr = pow(2, -args.lr)
  weight_decay = pow(2, -args.wd)
  optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

  dataloader_kwargs = {
  'batch_size': args.batch_size,
  'shuffle': False,   # We dont shuffle here as the train data is pre-shuffled.
  'num_workers': 1,
  'pin_memory': True,
  'pin_memory_device': "cuda:0"
  }
  dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

  # Minibatch prefetcher
  # batch_queue = queue.Queue(maxsize=args.buffer_size)
  # data_iter = iter(dataloader)
  # for _ in range(batch_queue.maxsize):
  #   try:
  #     X, y = next(data_iter)
  #     X = X.to(device)
  #     y = y.view(-1, 1).to(device)
  #     batch_queue.put((X, y))
  #   except StopIteration:
  #     break

  # Save model for every epoch
  epoch_data = []

  # Training
  logger.info(f"Started training ({args}) ...")
  model.train()
  total_start_event = torch.cuda.Event(enable_timing=True)
  total_end_event = torch.cuda.Event(enable_timing=True)
  total_start_event.record()

  epoch_start_events = []
  epoch_end_events = []

  epoch_weights = []
  epoch_bias = []

  for g_epoch in range(args.num_global_epochs):
    epoch_start_events.append(torch.cuda.Event(enable_timing=True))
    epoch_end_events.append(torch.cuda.Event(enable_timing=True))
    epoch_start_events[g_epoch].record()
    # for batch_id in range(len(dataloader)):
    for batch_id, (X, y) in enumerate(dataloader):
      X = X.to(device)
      y = y.to(device)
      # X, y = batch_queue.get()
      y_hat = model(X)
      loss = criterion(y_hat, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    epoch_end_events[g_epoch].record()
    # torch.cuda.synchronize()
    # epoch_time_ms = epoch_start_events[g_epoch].elapsed_time(epoch_end_event)
    # logger.info(f"Epoch {g_epoch} took {epoch_time_ms} ms. Processed {batch_id} batches")
    epoch_weights.append(deepcopy(model.linear.weight.data))
    epoch_bias.append(deepcopy(model.linear.bias.data))
    
  total_end_event.record()
  torch.cuda.synchronize()
  total_time_ms = total_start_event.elapsed_time(total_end_event)

  for g_epoch in range(args.num_global_epochs):
    epoch_time_ms = epoch_start_events[g_epoch].elapsed_time(epoch_end_events[g_epoch])
    epoch_data.append({
        "global_epoch_id": g_epoch,
        "model_weight": epoch_weights[g_epoch],
        "model_bias": epoch_bias[g_epoch],
        "total_time": epoch_time_ms,
    })
    logger.info(f"Epoch {g_epoch} took {epoch_time_ms} ms.")

  logger.info(f"Finished training.")
  torch.save({
      "args" : args,
      "epoch_data" : epoch_data,
      "total_training_time": total_time_ms,
    },
    f"{args.device}_{args.dataset}_{args.model_type}_{args.optim}_{args.num_global_epochs}_{args.batch_size}_{args.lr}_{args.wd}_{args.seed}.pt"
  )


def eval_CPU(rank, world_size, train_dataset, test_dataset, data_path, results_filename):
  df_lines = None
  logger = None
  if rank == 0:
    logger = setup_logger()
    df_lines = []

  # Initialize DDP
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '19244'
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

  result_files = []
  for root, dirs, files in os.walk(data_path):
    for file in files:
      if file.endswith(".pt"):
        result_files.append(file)

  for file in result_files:
    if rank == 0:
      logger.info(f"Evaluating {file}...")

    # Load all arguments during training
    data = torch.load(os.path.join(root, file), map_location=torch.device('cpu'))
    args = data["args"]


    # Model
    model = LinearModel(train_dataset.num_features)
    model = torch.nn.parallel.DistributedDataParallel(model)
    model.eval()
    
    # Loss
    criterion = None
    if train_dataset.model_type == "lr":
      criterion = nn.BCEWithLogitsLoss(reduction="sum")
    elif train_dataset.model_type == "svm":
      criterion = HingeLoss(margin=0.5, reduction="sum")
    else:
      raise RuntimeError(f"Unrecognized model type {train_dataset.model_type}!")

    weight_decay = pow(2, -args.wd)

    # Dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, world_size, rank, shuffle=False, seed=args.seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, world_size, rank, shuffle=False, seed=args.seed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)

    # Evaluation starts
    if rank == 0:
      logger.info(f"Started evaluation ({args}) ...")

    with model.no_sync():
      with torch.no_grad():
        for global_epoch_id in range(args.num_global_epochs):
          train_loss = 0.0
          train_correct = 0.0
          test_loss = 0.0
          test_correct = 0.0

          # Load the model parameters at this global epoch
          model.module.linear.weight.data = data["epoch_data"][global_epoch_id]["model_weight"]
          model.module.linear.bias.data = data["epoch_data"][global_epoch_id]["model_bias"]
          for batch_id, (X, y) in enumerate(train_dataloader):
            y_hat = model(X)
            train_loss += criterion(y_hat, y.view(-1, 1))
            if args.model_type == "lr":
              train_correct += ((torch.sigmoid(y_hat) >= 0.5).float() == y).float().sum()
            elif args.model_type == "svm":
              train_correct += (torch.sign(y_hat) == y).float().sum()
            else:
              raise RuntimeError(f"Unrecognized model {args.model_type}")
            
          for batch_id, (X, y) in enumerate(test_dataloader):
            y_hat = model(X)
            test_loss += criterion(y_hat, y.view(-1, 1))
            if args.model_type == "lr":
              test_correct += ((torch.sigmoid(y_hat) >= 0.5).float() == y).float().sum()
            elif args.model_type == "svm":
              test_correct += (torch.sign(y_hat) == y).float().sum()
            else:
              raise RuntimeError(f"Unrecognized model {args.model_type}")
            
          dist.all_reduce(train_loss.data, op=dist.ReduceOp.SUM)
          dist.all_reduce(train_correct.data, op=dist.ReduceOp.SUM)
          dist.all_reduce(test_loss.data, op=dist.ReduceOp.SUM)
          dist.all_reduce(test_correct.data, op=dist.ReduceOp.SUM)

          train_loss /= len(train_dataset)
          train_accuracy = 100 * (train_correct / len(train_dataset))
          test_loss /= len(test_dataset)
          test_accuracy = 100 * (test_correct / len(test_dataset))
          
          if rank == 0:
            model_norm = torch.linalg.norm(model.module.linear.weight)
            l2_reg = 0.5 * model_norm * model_norm * weight_decay

            # training_time = data["epoch_data"][global_epoch_id]["training_time"]
            # sync_time = data["epoch_data"][global_epoch_id]["sync_time"]
            total_time = data["epoch_data"][global_epoch_id]["total_time"]
            # logger.info(f"Epoch {global_epoch_id}: Train loss {train_loss}, acc {train_accuracy}%. Test loss {test_loss}, acc {test_accuracy}%. L2_reg {l2_reg}.")
            # logger.info(f"Epoch {global_epoch_id}: Total time {total_time}s (training {training_time}s, sync {sync_time}s)")          
            logger.info(f"Epoch {global_epoch_id}: Train loss {train_loss}, acc {train_accuracy}%. Test loss {test_loss}, acc {test_accuracy}%. L2_reg {l2_reg}.")
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
                train_loss.item(), train_accuracy.item(), test_loss.item(), test_accuracy.item(),
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
