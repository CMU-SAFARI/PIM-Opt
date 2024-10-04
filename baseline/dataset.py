import os, math

import torch
from torch.utils.data import Dataset
import torch.distributed as dist

import pandas as pd

from colored_logs import *


class YFCC100M(Dataset):   
  """
  Yahoo Flickr Creative Commons 100 Million (YFCC100M) Dataset
  Attr:
    - name (str): "YFCC100M"
    - model_type (str): The type of the model ("lr" or "svm"). If is SVM, label 0 will be converted to label -1
    - dataset_type (str): The type of the dataset ("train" or "test"). Only the .pt files with the same prefix as the dataset type will be loaded
    - num_features (int): The number of features per sample
  Args:
    - path (str): Path to the root directory that stores all the .pt files of the dataset
    - model_type (str): The type of the model ("lr" or "svm"). If is SVM, label 0 will be converted to label -1
    - dataset_type (str): The type of the dataset ("train" or "test"). Only the .pt files with the same prefix as the dataset type will be loaded
    - is_fast (bool): If True, only the first .pt file encountered will be loaded.
  """
  def __init__(self, path, dataset_size: int = -1, model_type="lr", dataset_type="train", is_fast=False, logger=None):
    self.name = "YFCC100M"
    self.num_features = 4096

    if model_type in ["lr", "svm"]:
      self.model_type = model_type
      logger.info(f"Model type: {model_type}")
    else:
      raise RuntimeError("Model type {model_type} not recognized. Supported types: \"lr\", \"svm\"") 
    
    if dataset_type in ["train", "test"]:
      self.dataset_type = dataset_type
    else:
      raise RuntimeError("Dataset type {dataset_type} not recognized. Supported types: \"train\", \"test\"") 

    if logger == None:
      logger = setup_logger()

    logger.info(f"Loading data from {path}...")
    selected_files = []
    try:
      for root, dirs, files in os.walk(path):
        for file in files:
          if file.startswith(self.dataset_type) and file.endswith(".pt"):
            selected_files.append(os.path.join(root, file))
            if is_fast:
              raise StopIteration
    except StopIteration:
      pass

    f = []
    print(selected_files)
    for file in selected_files:
      data = torch.load(file)

      # Turn label 0 into -1 for SVM
      if self.model_type == "svm":
        data[data[:, 0] == 0.0, 0] = -1.0

      f.append(data)

      logger.info(f"Loaded {os.path.basename(file)}")
    self.data = torch.cat(f, 0)
    if dataset_size != -1:
      self.data = self.data[:dataset_size, :]
    # Clean up to save memory
    for file in selected_files:
      del file
    logger.info(f"Dataset construction complete. Loaded {len(self.data)} samples.")

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx][1:], self.data[idx][:1]


class Criteo(Dataset):
  def __init__(self, path, model_type="lr", dataset_type="train", is_fast=False, logger=None):
    self.name = "CRITEO"
    self.num_features = 1000000

    if model_type in ["lr", "svm"]:
      self.model_type = model_type
      logger.info(f"Model type: {model_type}")
    else:
      raise RuntimeError("Model type {model_type} not recognized. Supported types: \"lr\", \"svm\"") 
    
    if dataset_type in ["train", "test"]:
      self.dataset_type = dataset_type
    else:
      raise RuntimeError("Dataset type {dataset_type} not recognized. Supported types: \"train\", \"test\"") 

    if logger == None:
      logger = setup_logger()

    logger.info(f"Loading data from {path}...")
    data_files = []
    try:
      for root, dirs, files in os.walk(path):
        for file in files:
          if file.startswith(self.dataset_type) and file.endswith(".pt"):
            data_files.append(os.path.join(root, file))
            if is_fast:
              raise StopIteration
    except StopIteration:
      pass

    _features = []
    for file in data_files:
      data = torch.load(file)
      _features.append(data.features)
    self.features = torch.cat(_features, 0)

    _labels = []
    for file in data_files:
      data = torch.load(file)
      if self.model_type == "svm":
        data.labels[data.labels[:, 0] == 0.0, 0] = -1.0
      _labels.append(data.labels)
    self.labels = torch.cat(_labels, 0)

    # Clean up to save memory
    for file in _features:
      del file
    for file in _labels:
      del file

    logger.info("Dataset construction complete.")


  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]


class CriteoAlt(Dataset):
  def __init__(self, path, model_type="lr", dataset_type="train", is_fast=False, unstack=False, logger=None):
    self.name = "CRITEO"
    self.num_features = 1000000

    if model_type in ["lr", "svm"]:
      self.model_type = model_type
      logger.info(f"Model type: {model_type}")
    else:
      raise RuntimeError("Model type {model_type} not recognized. Supported types: \"lr\", \"svm\"") 
    
    if dataset_type in ["train", "test"]:
      self.dataset_type = dataset_type
    else:
      raise RuntimeError("Dataset type {dataset_type} not recognized. Supported types: \"train\", \"test\"") 

    if logger == None:
      logger = setup_logger()

    logger.info(f"Loading data from {path}...")

    chunk = pd.read_csv(path, index_col=None, header=None, delim_whitespace=True)  
    data = torch.tensor(chunk.values, dtype=torch.int32)
    labels = data[:, 0]
    features = data[:, 1:]

    indices_rows = torch.repeat_interleave(torch.arange(features.size()[0], dtype=torch.int32), features.size()[1])
    indices_columns = features.flatten()

    if not unstack:
      sparse_tensor = torch.sparse_coo_tensor(torch.stack([indices_rows, indices_columns]), torch.ones(indices_columns.shape[0]), torch.Size([features.size()[0], 1000000]))
      self.features = sparse_tensor
      self.is_ready = True
    else:
      self.features_idx = torch.stack([indices_rows, indices_columns])
      self.features_val = torch.ones(indices_columns.shape[0])
      self.is_ready = False

    if self.model_type == "svm":
      labels[labels[:, 0] == 0.0, 0] = -1.0

    self.labels = labels

    logger.info("Dataset construction complete.")


  def stack_features(self):
    self.features = torch.sparse_coo_tensor(self.features_idx, self.features_val, torch.Size([len(self.labels), 1000000]))
    self.is_ready = True

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]


class CriteoDense(Dataset):
  def __init__(self, path, model_type="lr", dataset_type="train", is_fast=False, logger=None):
    self.name = "CRITEO"
    self.num_features = 1000000

    if model_type in ["lr", "svm"]:
      self.model_type = model_type
      logger.info(f"Model type: {model_type}")
    else:
      raise RuntimeError("Model type {model_type} not recognized. Supported types: \"lr\", \"svm\"") 
    
    if dataset_type in ["train", "test"]:
      self.dataset_type = dataset_type
    else:
      raise RuntimeError("Dataset type {dataset_type} not recognized. Supported types: \"train\", \"test\"") 

    # if logger == None:
    #   logger = setup_logger()

    logger.info(f"Loading data from {path}...")
    data_files = []
    try:
      for root, dirs, files in os.walk(path):
        for file in files:
          if file.startswith(self.dataset_type) and file.endswith(".pt"):
            data_files.append(os.path.join(root, file))
            if is_fast:
              raise StopIteration
    except StopIteration:
      pass

    _features = []
    _labels = []

    for i, file in enumerate(data_files):
      data = torch.load(file)
      _features.append(data[:, 1:])
      _labels.append(data[:, :1].float())
      # logger.info(f"Loading {file}...")
    self.features = torch.cat(_features, 0)
    self.labels = torch.cat(_labels, 0)

    if self.model_type == "svm":
      self.labels[self.labels[:, 0] == 0.0, 0] = -1.0

    # Clean up to save memory
    for file in _features:
      del file
    for file in _labels:
      del file

    logger.info("Dataset construction complete.")


  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]


class CriteoDenseGPU(Dataset):
  def __init__(self, path, dataset_size: int, model_type="lr", dataset_type="train", is_fast=False, logger=None):
    self.name = "CRITEO"
    self.num_features = 1000000

    if model_type in ["lr", "svm"]:
      self.model_type = model_type
      logger.info(f"Model type: {model_type}")
    else:
      raise RuntimeError("Model type {model_type} not recognized. Supported types: \"lr\", \"svm\"") 
    
    if dataset_type in ["train", "test"]:
      self.dataset_type = dataset_type
    else:
      raise RuntimeError("Dataset type {dataset_type} not recognized. Supported types: \"train\", \"test\"") 

    # if logger == None:
    #   logger = setup_logger()
    df = pd.read_csv(path, index_col=None, header=None, delim_whitespace=True)
    data = torch.tensor(df.values, dtype=torch.int32)
    data = data[:dataset_size, :]

    self.features = data[:, 1:]
    self.labels = data[:, :1].float()

    if self.model_type == "svm":
      self.labels[self.labels[:, 0] == 0.0, 0] = -1.0

    del df

    logger.info(f"Dataset construction complete ({len(self.labels)}).")


  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]


class CriteoDistributed(Dataset):
  def __init__(self, path, model_type="lr", dataset_type="train", is_fast=False, logger=None):
    self.name = "CRITEO"
    self.num_features = 1000000

    self.path = path


    self.logger = logger

    if model_type in ["lr", "svm"]:
      self.model_type = model_type
      self.logger.info(f"Model type: {model_type}")
    else:
      raise RuntimeError("Model type {model_type} not recognized. Supported types: \"lr\", \"svm\"") 
    
    if dataset_type in ["train", "test"]:
      self.dataset_type = dataset_type
    else:
      raise RuntimeError("Dataset type {dataset_type} not recognized. Supported types: \"train\", \"test\"") 

  def load_rank_data(self, world_size: int, rank: int, batch_size: int, dataset_size: int):
    print(f"World size {world_size}, Rank {rank}, Batch Size {batch_size}, Dataset Size {dataset_size}")
    print(f"Loading data from {self.path}...")

    chunk_size = batch_size * world_size
    num_chunks = math.floor(dataset_size / chunk_size)

    batch_data = []
    for chunk_idx in range(num_chunks):
      num_skiprows = chunk_idx * chunk_size + rank * batch_size
      batch_data.append(pd.read_csv(self.path, index_col=None, header=None, delim_whitespace=True, skiprows=num_skiprows, nrows=batch_size))
    rank_df = pd.concat(batch_data)
    data = torch.tensor(rank_df.values, dtype=torch.int32)
    labels = data[:, 0]
    features = data[:, 1:]

    indices_rows = torch.repeat_interleave(torch.arange(features.size()[0], dtype=torch.int32), features.size()[1])
    indices_columns = features.flatten()

    sparse_tensor = torch.sparse_coo_tensor(torch.stack([indices_rows, indices_columns]), torch.ones(indices_columns.shape[0]), torch.Size([features.size()[0], 1000000]))
    self.features = sparse_tensor

    if self.model_type == "svm":
      labels[labels[:, 0] == 0.0, 0] = -1.0

    self.labels = labels

    print(f"Rank {rank} dataset construction complete.")


  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]


class CriteoDenseDistributed(Dataset):
  def __init__(self, path, model_type="lr", dataset_type="train", is_fast=False, logger=None):
    self.name = "CRITEO"
    self.num_features = 1000000
    self.path = path

    if model_type in ["lr", "svm"]:
      self.model_type = model_type
      logger.info(f"Model type: {model_type}")
    else:
      raise RuntimeError("Model type {model_type} not recognized. Supported types: \"lr\", \"svm\"") 
    
    if dataset_type in ["train", "test"]:
      self.dataset_type = dataset_type
    else:
      raise RuntimeError("Dataset type {dataset_type} not recognized. Supported types: \"train\", \"test\"") 

    if logger == None:
      logger = setup_logger()
    else:
      self.logger = logger


  def load_rank_data(self, world_size: int, rank: int, batch_size: int, dataset_size: int):
    if rank == 0:
      self.logger.info(f"World size {world_size}, Batch Size {batch_size}, Dataset Size {dataset_size}")
      self.logger.info(f"Loading data from {self.path}...")
      rank_data_list = []
      for i in range(world_size):
        rank_data_list.append([])

      block_size = 64
      chunk_size = batch_size * world_size
      num_chunks = int(dataset_size / chunk_size)
      num_blocks = int(num_chunks / block_size)

      if rank == 0:
        self.logger.info(f"Block size {block_size}, Num Blocks {num_blocks}, Chunk size {chunk_size}, Num Chunks {num_chunks}")

      for i, chunk in enumerate(pd.read_csv(self.path, index_col=None, header=None, chunksize=chunk_size * block_size, delim_whitespace=True)):
        self.logger.info(f"   Processing block {i}/{num_blocks-1}...")
        data = torch.tensor(chunk.values, dtype=torch.int32)
        split_data = torch.split(data, batch_size)
        for j in range(world_size):
          for k in range(block_size):
            rank_data_list[j].append(split_data[j+ k * block_size])

        if i == num_blocks - 1:
          break

      concat_rank_data_list = []
      for i in range(world_size):
        concat_rank_data_list.append(torch.cat(rank_data_list[i]))

      rank_data = torch.empty((int(dataset_size / world_size), 40), dtype=torch.int32)
      dist.scatter(rank_data, concat_rank_data_list, src=0)
    else:
      rank_data = torch.empty((int(dataset_size / world_size), 40), dtype=torch.int32)
      dist.scatter(rank_data, None, src=0)

    if self.model_type == "svm":
      rank_data[rank_data[:, 0] == 0.0, 0] = -1.0

    labels = rank_data[:, 0].float()
    self.labels = labels

    features = rank_data[:, 1:]
    self.features = features

    if rank == 0:
      self.logger.info(f"Dataset construction and distribution complete")
      self.logger.info(f"Rank data size {len(self.labels)}")
      

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]


class CriteoScatter(Dataset):
  def __init__(self, path, model_type="lr", dataset_type="train", is_fast=False, logger=None):
    self.name = "CRITEO"
    self.num_features = 1000000

    self.path = path
    self.logger = logger

    if model_type in ["lr", "svm"]:
      self.model_type = model_type
      self.logger.info(f"Model type: {model_type}")
    else:
      raise RuntimeError("Model type {model_type} not recognized. Supported types: \"lr\", \"svm\"") 
    
    if dataset_type in ["train", "test"]:
      self.dataset_type = dataset_type
    else:
      raise RuntimeError("Dataset type {dataset_type} not recognized. Supported types: \"train\", \"test\"") 



  def load_rank_data(self, world_size: int, rank: int, batch_size: int, dataset_size: int):
    if rank == 0:
      self.logger.info(f"World size {world_size}, Batch Size {batch_size}, Dataset Size {dataset_size}")
      self.logger.info(f"Loading data from {self.path}...")
      rank_data_list = []
      for i in range(world_size):
        rank_data_list.append([])

      chunk_size = batch_size * world_size
      num_chunks = int(dataset_size / chunk_size)
      for i, chunk in enumerate(pd.read_csv(self.path, index_col=None, header=None, chunksize=chunk_size, delim_whitespace=True)):
        data = torch.tensor(chunk.values, dtype=torch.float32)
        split_data = torch.split(data, batch_size)
        self.logger.info(f"   Processing chunk {i}/{num_chunks-1}...")
        for j in range(world_size):
          rank_data_list[j].append(split_data[j])

        if i == num_chunks - 1:
          break

      concat_rank_data_list = []
      for i in range(world_size):
        concat_rank_data_list.append(torch.cat(rank_data_list[i]))

      rank_data = torch.empty((int(dataset_size / world_size), 40), dtype=torch.float32)
      dist.scatter(rank_data, concat_rank_data_list, src=0)
    else:
      rank_data = torch.empty((int(dataset_size / world_size), 40), dtype=torch.float32)
      dist.scatter(rank_data, None, src=0)

    labels = rank_data[:, 0]
    features = rank_data[:, 1:]

    indices_rows = torch.repeat_interleave(torch.arange(features.size()[0], dtype=torch.int32), features.size()[1])
    indices_columns = features.flatten()

    sparse_tensor = torch.sparse_coo_tensor(torch.stack([indices_rows, indices_columns]), torch.ones(indices_columns.shape[0]), torch.Size([features.size()[0], 1000000]))
    self.features = sparse_tensor

    if self.model_type == "svm":
      labels[labels[:, 0] == 0.0, 0] = -1.0

    self.labels = labels

    if rank == 0:
      self.logger.info(f"Dataset construction and distribution complete")


  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]


class Higgs(Dataset):   
  def __init__(self, path, model_type="lr", dataset_type="train", logger=None):
    self.name = "Higgs"
    self.num_features = 28

    if model_type in ["lr", "svm"]:
      self.model_type = model_type
      logger.info(f"Model type: {model_type}")
    else:
      raise RuntimeError("Model type {model_type} not recognized. Supported types: \"lr\", \"svm\"") 
    
    if dataset_type in ["train", "test"]:
      self.dataset_type = dataset_type
    else:
      raise RuntimeError("Dataset type {dataset_type} not recognized. Supported types: \"train\", \"test\"") 

    if logger == None:
      logger = setup_logger()

    logger.info(f"Loading data from {path}...")
    selected_files = []
    for root, dirs, files in os.walk(path):
      for file in files:
        if file.startswith(self.dataset_type) and file.endswith(".pt"):
          selected_files.append(os.path.join(root, file))

    f = []
    for file in selected_files:
      data = torch.load(file)

      # Turn label 0 into -1 for SVM
      if self.model_type == "svm":
        data[data[:, 0] == 0.0, 0] = -1.0

      f.append(data)

      logger.info(f"Loaded {os.path.basename(file)}")
    self.data = torch.cat(f, 0)

    # Clean up to save memory
    for file in selected_files:
      del file
    logger.info("Dataset construction complete.")
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx][1:], self.data[idx][:1]
