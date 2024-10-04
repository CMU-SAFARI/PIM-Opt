import os
import argparse
import random

import numpy as np

import torch.multiprocessing as mp

from colored_logs import *
from model import *
from smodel import *
from dataset import *


def main():
  logger = setup_logger()

  parser = argparse.ArgumentParser(description='Main script for UPMEM ML')
  parser.add_argument('--model', type=str, default="lr", metavar='S',
                      help='model type (\"lr\" or \"svm\")')
  parser.add_argument('--dataset', type=str, default="yfcc", metavar='S',
                      help='the dataset to eval with')
  parser.add_argument('--path', type=str, required=True, metavar='S',
                      help='path to the root directory containing the dataset')
  parser.add_argument('--epoch_data_path', type=str, required=True, metavar='S',
                      help='path to the root directory storing the data for every epoch')
  parser.add_argument('--output', type=str, required=True, metavar='S',
                      help='output CSV filename')
  parser.add_argument('--num_procs', type=int, default=128, metavar='N',
                      help='number of processes to spawn for the evaluation')
  args = parser.parse_args()

  print(args)

  train_dataset = None
  test_dataset = None
  if args.dataset == "yfcc":
    train_dataset = YFCC100M(path=args.path, model_type=args.model, dataset_type="train", is_fast=False, logger=logger)
    test_dataset = YFCC100M(path=args.path, model_type=args.model, dataset_type="test", is_fast=False, logger=logger)
    eval_func = eval_CPU
    mp.spawn(eval_func,
        args=(args.num_procs, train_dataset, test_dataset, args.epoch_data_path, args.output),
        nprocs=args.num_procs,
        join=True)
  elif args.dataset == "criteo":
    criteo_train_dataset_path = args.path + "Criteo_train_NR_DPUS_2048_label_0_388882440_label_1_13770744_total_402653184.txt"
    criteo_test_dataset_path = args.path + "test_data_criteo_tb_day_23.txt"
    train_dataset = CriteoDenseDistributed(path=criteo_train_dataset_path, model_type=args.model, dataset_type="train", is_fast=False, logger=logger)
    test_dataset = CriteoDenseDistributed(path=criteo_test_dataset_path, model_type=args.model, dataset_type="test", is_fast=False, logger=logger)
    eval_func = eval_CPU_sparse
    mp.spawn(eval_func,
            args=(args.num_procs, train_dataset, 402653184, test_dataset, 176160768, args.epoch_data_path, args.output),
            nprocs=args.num_procs,
            join=True)
  else:
    raise RuntimeError(f"Dataset {args.dataset} unrecognized!")           

if __name__ == "__main__":
  main()
