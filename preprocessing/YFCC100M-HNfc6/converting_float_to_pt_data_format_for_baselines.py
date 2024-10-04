import time
import sys

import pandas as pd
import numpy as np
import torch



if __name__ == "__main__":
  source_path_prefix = str(sys.argv[1])
  train_path = source_path_prefix + "preprocessed_datasets/YFCC100M-HNfc6/float/YFCC100M_train_corresponding_to_NR_DPUs_2048_label_0_3407872_label_1_3407872_total_6815744_float.txt"
  test_path = source_path_prefix + "preprocessed_datasets/YFCC100M-HNfc6/float/YFCC100M_test_corresponding_to_NR_DPUs_2048_label_0_851968_label_1_851968_total_1703936_float.txt"
  files = {
    "train" : train_path,
    "test" : test_path,    
  }

  for dataset_type in files:
    for i, chunk in enumerate(pd.read_csv(files[dataset_type], index_col=None, header=None, chunksize=5000000, delim_whitespace=True)):
      print(f"Processing chunk {i} for {dataset_type} dataset")
      data = torch.tensor(chunk.values, dtype=torch.float32)
      save_path = source_path_prefix + "preprocessed_datasets/YFCC100M-HNfc6/float/" + "pytorch_data/{}.{}.pt".format(dataset_type, i)
      torch.save(data, save_path)