import os
import argparse
import random

import numpy as np

import torch.multiprocessing as mp

from colored_logs import *
from model import *
from smodel import *
from dataset import *

train_params = {
  "yfcc" : {
    "gpu" : {
      "lr" : {
        "sgd" : {
          "batch_size" : [4096],
          "learning_rate": [4],
          "weight_decay": [10],
          "alpha": [-1]
        },
      }, 

      "svm" : {
        "sgd" : {
          "batch_size" : [4096],
          "learning_rate": [6],
          "weight_decay": [8],
          "alpha": [-1]
        },
      }, 
    }
  },
}


def main():
  logger = setup_logger()

  parser = argparse.ArgumentParser(description='Main script for UPMEM ML')
  parser.add_argument('--device', type=str, default="cpu", metavar='S',
                      help='device type (\"cpu\" or \"gpu\")')

  parser.add_argument('--dataset', type=str, default="yfcc", metavar='S',
                      help='the dataset to train with')
  parser.add_argument('--path', type=str, required=True, metavar='S',
                      help='the path to the dataset')
  parser.add_argument('--dataset_size', type=int, default=-1, metavar='S',
                      help='number of samples in the training data (default: -1, only relevant for CRITEO)')

  parser.add_argument('--model', type=str, default="lr", metavar='S',
                      help='model type (\"lr\" or \"svm\")')
  parser.add_argument('--optim', type=str, default="sgd", metavar='S',
                      help='optimizer type (\"sgd\" or \"admm\")')
  parser.add_argument('--dist_type', type=str, default="ma", metavar='S',
                      help='distributed training type (\"ma\" or \"ga\")')

  parser.add_argument('--fast', action='store_true',
                      help='Use only one partition of the dataset for a quick test run')



  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  cmdl_args = parser.parse_args()

  train_args = TrainArgs()
  train_args.device = cmdl_args.device
  train_args.dataset = cmdl_args.dataset
  train_args.dataset_size = cmdl_args.dataset_size
  train_args.model_type = cmdl_args.model
  train_args.optim = cmdl_args.optim
  train_args.seed = cmdl_args.seed
  train_args.dist_type = cmdl_args.dist_type

  # Fix random seeds for reproducibility
  torch.manual_seed(train_args.seed)
  random.seed(train_args.seed)
  np.random.seed(train_args.seed)
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # Needed for CUDA
  torch.use_deterministic_algorithms(True)
  torch.set_default_dtype(torch.float32)

  # Choose the train function
  train_func = None
  if cmdl_args.device == "cpu":
    train_func = train_CPU
    if cmdl_args.dataset == "criteo":
      train_func = train_CPU_sparse
  elif cmdl_args.device == "gpu":
    train_func = train_GPU
  else:
    raise RuntimeError(f"Unknown device \"{cmdl_args.device}\"!")

  train_dataset = None
  if cmdl_args.dataset == "yfcc":
    train_dataset = YFCC100M(path=cmdl_args.path, dataset_size=cmdl_args.dataset_size, model_type=cmdl_args.model, dataset_type="train", is_fast=cmdl_args.fast, logger=logger)
  elif cmdl_args.dataset == "criteo":
    train_dataset = CriteoDenseDistributed(path=cmdl_args.path, model_type=cmdl_args.model, dataset_type="train", logger=logger)
  elif cmdl_args.dataset == "higgs":
    train_dataset = Higgs(path=cmdl_args.path, model_type=cmdl_args.model, dataset_type="train", logger=logger)
  else:
    raise RuntimeError(f"Dataset {cmdl_args.dataset} unrecognized!")


  for num_global_epochs in [10]:
    for num_local_epochs in [1]:
      for model in train_params[train_args.dataset][train_args.device]:
        for optim in train_params[train_args.dataset][train_args.device][model]:
          for batch_size in train_params[train_args.dataset][train_args.device][model][optim]["batch_size"]:
            for lr in train_params[train_args.dataset][train_args.device][model][optim]["learning_rate"]:
              for wd in train_params[train_args.dataset][train_args.device][model][optim]["weight_decay"]:
                for alpha in train_params[train_args.dataset][train_args.device][model][optim]["alpha"]:
                  print(f"{model} {optim} {batch_size} {lr} {wd} {alpha}")
                  train_args.model_type = model
                  train_args.num_global_epochs = num_global_epochs
                  train_args.num_local_epochs = num_local_epochs
                  train_args.batch_size = batch_size
                  train_args.lr = lr
                  train_args.wd = wd
                  train_args.alpha = alpha
                  train_args.num_procs = 1


                  print(train_args)

                  if cmdl_args.device == "cpu":
                    mp.spawn(train_func,
                            args=(train_dataset, train_args),
                            nprocs=train_args.num_procs,
                            join=True)
                  else:
                    train_func(train_dataset, train_args, logger)

if __name__ == "__main__":
  main()
