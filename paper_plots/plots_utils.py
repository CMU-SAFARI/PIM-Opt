import os, sys

import seaborn as sns
import numpy as np
import pandas as pd

import importlib




def process_DPU_csv(path):
  dpu_df = pd.read_csv(path)
  dpu_df = dpu_df.drop('epoch_communication_time', axis = 1)
  dpu_df = dpu_df.drop('epoch_compute_time', axis = 1)
  dpu_df.loc[dpu_df['algorithm'] == 'mbsgd', 'reg_term_alpha'] = 0
  dpu_df.loc[dpu_df['algorithm'] == 'ga', 'reg_term_alpha'] = 0
  dpu_df.loc[dpu_df['algorithm'] == 'mbsgd', 'algorithm'] = 'MA-SGD'
  dpu_df.loc[dpu_df['algorithm'] == 'ga', 'algorithm'] = 'GA-SGD'
  dpu_df.loc[dpu_df['algorithm'] == 'admm', 'algorithm'] = 'ADMM'
  dpu_df.loc[dpu_df['architecture'] == 'dpu', 'architecture'] = 'DPU'
  return dpu_df


def process_benchmark_DPU_csv(path):
    dpu_df = pd.read_csv(path)
    dpu_df = dpu_df.drop('num_global_epochs', axis = 1)
    dpu_df.loc[dpu_df['algorithm'] == 'mbsgd', 'reg_term_alpha'] = 0
    dpu_df.loc[dpu_df['algorithm'] == 'ga', 'reg_term_alpha'] = 0
    dpu_df.loc[dpu_df['algorithm'] == 'mbsgd', 'algorithm'] = 'MA-SGD'
    dpu_df.loc[dpu_df['algorithm'] == 'ga', 'algorithm'] = 'GA-SGD'
    dpu_df.loc[dpu_df['algorithm'] == 'admm', 'algorithm'] = 'ADMM'
    dpu_df.loc[dpu_df['architecture'] == 'dpu', 'architecture'] = 'DPU'
    return dpu_df

def process_baseline_csv(path):
  df_baseline = pd.read_csv(path)
  df_baseline.loc[df_baseline['algorithm'] == 'mbsgd', 'reg_term_alpha'] = 0
  df_baseline = df_baseline.rename(columns={'num_procs': 'nr_procs'})

  df_baseline = df_baseline.drop(["seed", "l2_reg"], axis=1)

  df_baseline['scaling_type'] = 'weak'
  df_baseline['start_up_time'] = 0
  df_baseline.loc[(df_baseline['algorithm'] == 'mbsgd') & (df_baseline['dist_type'] == 'ma'), 'algorithm'] = 'MA-SGD'
  df_baseline.loc[(df_baseline['algorithm'] == 'mbsgd') & (df_baseline['dist_type'] == 'ga'), 'algorithm'] = 'GA-SGD'
  df_baseline.loc[df_baseline['architecture'] == 'GPU', 'algorithm'] = 'SGD'
  df_baseline.loc[df_baseline['algorithm'] == 'admm', 'algorithm'] = 'ADMM'
  df_baseline.loc[(df_baseline['architecture'] == 'GPU') & ((df_baseline['algorithm'] == 'GA-SGD')), 'num_local_epochs'] = 1
  df_baseline.loc[(df_baseline['architecture'] == 'GPU') & ((df_baseline['algorithm'] == 'GA-SGD')), 'epoch_time'] /= 1000.0 # convert ms to s
  df_baseline.loc[(df_baseline['architecture'] == 'CPU') & ((df_baseline['algorithm'] == 'GA-SGD')), 'batch_size'] *=128
  df_baseline.loc[df_baseline['architecture'] == 'GPU', 'num_local_epochs'] = 1
  df_baseline.loc[df_baseline['architecture'] == 'GPU', 'nr_procs'] = 1
  df_baseline = df_baseline[(df_baseline['nr_procs'] == 128) | (df_baseline['architecture'] != 'CPU')]

  df_baseline = df_baseline.drop(["dist_type"], axis=1)
  df_baseline = df_baseline.groupby([
    "architecture", "dataset", "model_type", "algorithm", "type_precision",
    "num_global_epochs", "num_local_epochs", "batch_size", "lr", "reg_term", "reg_term_alpha",
    "nr_procs"
    ]).apply(lambda group: group.sort_values(by='g_epoch_id')).reset_index(drop=True)
  df_baseline['total_elapsed_time'] = df_baseline.groupby([
    "architecture", "dataset", "model_type", "algorithm", "type_precision",
    "num_global_epochs", "num_local_epochs", "batch_size", "lr", "reg_term", "reg_term_alpha",
    "nr_procs"
    ])['epoch_time'].cumsum()

  

  
  
  return df_baseline


def find_txt_filenames(root_folder):
    txt_files = []
    file_handles = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        if dirpath.split(os.sep)[-1].startswith(("admm", "mbsgd", "ga", "gama")):
            for filename in filenames:
                if filename.endswith(".txt"):
                    txt_files.append(filename)
                    filepath = os.path.join(dirpath, filename)
                    file_handle = open(filepath, 'r')
                    file_handles.append(file_handle)
    return txt_files, file_handles



def find_txt_filenames_new(root_folder):
    filenames_list = []
    filepaths_list = []

    for dirpath, dirnames, files in os.walk(root_folder):
        if dirpath.split(os.sep)[-1].startswith(("admm", "mbsgd", "ga", "gama")):
            for file in files:
                if file.endswith(".txt"):
                    filenames_list.append(file)
                    filepaths_list.append(os.path.join(dirpath, file))
    return filenames_list, filepaths_list



def parse_filename(tag, filename):
    x = 0
    splits = filename.split('__')
    if tag == 'reg_term':
        for split in splits:
            if tag in split:
                if not ('reg_term_alpha' in split):
                    sub_splits = split.split('_')
                    x = np.int_(sub_splits[len(sub_splits)-1])
    else:
        for split in splits:
            if tag in split:
                sub_splits = split.split('_')
                if tag == 'learning_rate':
                    x = np.int_(sub_splits[len(sub_splits)-1])
                elif tag == 'alpha':
                    x = np.int_(sub_splits[len(sub_splits)-1])
                elif tag == 'NR_DPUS':
                    x = np.int_(sub_splits[len(sub_splits)-1])
                elif tag == 'epochs':
                    x = np.int_(sub_splits[len(sub_splits)-1])
                    break
                elif tag == 'b_size_frac':
                    x = np.int_(sub_splits[len(sub_splits)-1])

                elif tag == 'num_local_epochs':
                    sub_splits = sub_splits[len(sub_splits)-1].split('.')[0]
                    x = np.int_(sub_splits[len(sub_splits)-1])
                else:
                    print('tag not well-defined!')
    return x

def parse_filename_local_epochs(filename):
    splits = filename.split('__')
    splits = splits[len(splits)-1]
    splits = splits.split('.')
    splits = splits[0]
    splits = splits.split('_')
    x = np.int_(splits[len(splits)-1])
    return x

def save_csv_UPMEM(save_path, filenames_list, file_paths_list):
    create_df = pd.DataFrame(columns=['scaling_type', 'architecture', 'type_precision', 'algorithm', 'nr_procs', 'batch_size', 'lr', 'reg_term', 'reg_term_alpha', 'num_global_epochs', 'num_local_epochs', 'start_up_time', 'total_elapsed_time', 'epoch_time', 'epoch_compute_time', 'epoch_communication_time', 'g_epoch_id', 'train_accuracy', 'test_accuracy', 'train_loss', 'test_loss'])

    for k in range(0, len(filenames_list)):

        filename_path = filenames_list[k]
        filename = filenames_list[k].split('/')[-1]
        file_path = file_paths_list[k]
        architecture = 'DPU'
        dataset = ''
        if 'YFCC' in file_path:
            dataset = 'yfcc'
        elif 'Criteo' in file_path:
            dataset = 'criteo'


        scaling_type = ''
        if 'hyperparameter_search' in file_path:
            continue
        elif 'strong_scaling' in file_path:
            scaling_type = 'strong'
        else:
            scaling_type = 'weak'


        model_type = ''
        if 'lr_' in filename:
            model_type = 'lr'
        elif 'svm_' in filename:
            model_type = 'svm'
        else:
            print('Filename: {} has no model_type!!!\n'.format(filename))

        type_precision = 'uint32'

        
        algorithm = ''
        if 'admm' in file_path:
            algorithm = 'admm'
        elif 'mbsgd' in file_path:
            algorithm = 'mbsgd'
        elif 'ga_' in file_path:
            algorithm = 'ga'
        
        nr_procs = parse_filename('NR_DPUS', filename)
        batch_size = parse_filename('b_size_frac', filename)
        lr = parse_filename('learning_rate', filename)
        reg_term = parse_filename('reg_term', filename)
        if algorithm == 'admm':
            reg_term_alpha = parse_filename('alpha', filename)
        else:
            reg_term_alpha = np.nan
        num_global_epochs = parse_filename('epochs', filename)
        num_local_epochs = parse_filename_local_epochs(filename)


        df_tmp = pd.DataFrame(columns=['scaling_type', 'dataset', 'architecture', 'model_type', 'type_precision', 'algorithm', 'nr_procs', 'batch_size', 'lr', 'reg_term', 'reg_term_alpha', 'num_global_epochs', 'num_local_epochs', 'start_up_time', 'total_elapsed_time', 'epoch_time', 'epoch_compute_time', 'epoch_communication_time', 'g_epoch_id'])


        df_tmp_compute_error = pd.DataFrame(columns=['g_epoch_id', 'train_accuracy', 'test_accuracy', 'train_loss', 'test_loss'])

        file_tmp = open(file_path, 'r')

        start_successful = False  
        end_successful = False
        while (True):
            start_successful 
            line = file_tmp.readline()
            if not line:
                break
            if 'Experiment successfully started.' in line:
                start_successful = True
            if 'Experiment successfully completed.' in line:
                end_successful = True
        

        if (not start_successful) and (not end_successful):
            print('filename: {}\n'.format(filename))
            break
        file_tmp.close()
        file = open(file_path, 'r')
        start_up_time = 0.0
        epoch_compute_time = 0.0
        epoch_communication_time = 0.0
        epoch_time = 0.0
        total_elapsed_time = 0.0
        train_accuracy = 0.0
        test_accuracy = 0.0
        train_loss = 0.0
        test_loss = 0.0
        g_epoch_id = 0
        compute_error_rate = False
        while (True):
            line = file.readline()
            if not line:
                break
            if ('Gathering data on compute error and others' in line) or ('Gathering data on compute accuracy and others' in line):
                compute_error_rate = True
            if (not compute_error_rate):
                if 'elapsed time for allocation of DPUs' in line:
                    splits = line.split(' ')
                    load_binary_to_DPU = np.float_(splits[len(splits)-2])
                    start_up_time += load_binary_to_DPU
                if 'Preprocessing' in line:
                    splits = line.split(' ')
                    preprocessing = np.float_(splits[len(splits)-2])
                    start_up_time += preprocessing
                if 'TransPimLib' in line:
                    splits = line.split(' ')
                    tmp = np.float_(splits[len(splits)-2])
                    start_up_time += tmp
                if 'Load input data to DPUs' in line:
                    splits = line.split(' ')
                    load_input_data_to_DPUs = np.float_(splits[len(splits)-2])
                    start_up_time += load_input_data_to_DPUs
                if 'to DPUs. Elapsed time is' in line:
                    splits = line.split(' ')
                    load_model_to_dpus = np.float_(splits[len(splits)-2])
                    total_elapsed_time += load_model_to_dpus
                    epoch_communication_time += load_model_to_dpus
                    epoch_time += load_model_to_dpus
                if 'DPU kernel time' in line:
                    splits = line.split(' ')
                    dpu_kernel_time = np.float_(splits[len(splits)-2])
                    total_elapsed_time += dpu_kernel_time
                    epoch_compute_time += dpu_kernel_time
                    epoch_time += dpu_kernel_time
                if 'Retrieve the models of all the DPUs.' in line:  
                    splits = line.split(' ')
                    retrieve_model_from_dpus = np.float_(splits[len(splits)-2])
                    total_elapsed_time += retrieve_model_from_dpus
                    epoch_communication_time += retrieve_model_from_dpus
                    epoch_time += retrieve_model_from_dpus
                if ('odel averaging' in line) or ('radient averaging' in line): 
                    splits = line.split(' ')
                    model_averaging = np.float_(splits[len(splits)-2])
                    total_elapsed_time += model_averaging
                    epoch_communication_time += model_averaging
                    epoch_time += model_averaging
                    g_epoch_id = np.int_(splits[1].split('.')[0])
                    row = pd.Series([scaling_type, dataset, architecture, model_type, type_precision, algorithm, nr_procs, batch_size, lr, reg_term, reg_term_alpha, num_global_epochs, num_local_epochs, start_up_time, total_elapsed_time, epoch_time, epoch_compute_time, epoch_communication_time, g_epoch_id], index=df_tmp.columns)
                    df_tmp.loc[len(df_tmp)] = row
                    df_tmp.reset_index()
                    if nr_procs == 256:
                        if scaling_type == 'weak':
                            scaling_type = 'strong'
                            row = pd.Series([scaling_type, dataset, architecture, model_type, type_precision, algorithm, nr_procs, batch_size, lr, reg_term, reg_term_alpha, num_global_epochs, num_local_epochs, start_up_time, total_elapsed_time, epoch_time, epoch_compute_time, epoch_communication_time, g_epoch_id], index=df_tmp.columns)
                            df_tmp.loc[len(df_tmp)] = row
                            df_tmp.reset_index()
                            scaling_type = 'weak'

                    epoch_time = 0.0
                    epoch_compute_time = 0.0
                    epoch_communication_time = 0.0
                else:
                    continue
            else:
                if 'Training accuracy of averaged model' in line:
                    splits = line.split(',')
                    splits = splits[0]
                    splits = splits.split(' ')
                    train_accuracy = np.float_(splits[len(splits)-1])
                if 'Train. roc_auc_score' in line:
                    splits = line.split('=')
                    train_accuracy = np.float_(splits[len(splits)-1])
                if 'Test accuracy of averaged model' in line:
                    splits = line.split(',')
                    splits = splits[0]
                    splits = splits.split(' ')
                    test_accuracy = np.float_(splits[len(splits)-1])
                if 'Test. roc_auc_score' in line:
                    splits = line.split('=')
                    test_accuracy = np.float_(splits[len(splits)-1])
                    row = pd.Series([g_epoch_id, train_accuracy, test_accuracy, train_loss, test_loss], index=df_tmp_compute_error.columns)
                    df_tmp_compute_error.loc[len(df_tmp_compute_error)] = row
                    df_tmp_compute_error.reset_index()
                    if nr_procs == 256:
                        if scaling_type == 'weak':
                            scaling_type = 'strong'
                            row = pd.Series([g_epoch_id, train_accuracy, test_accuracy, train_loss, test_loss], index=df_tmp_compute_error.columns)
                            df_tmp_compute_error.loc[len(df_tmp_compute_error)] = row
                            df_tmp_compute_error.reset_index()
                            scaling_type = 'weak'
                if ('Training cross entropy loss' in line) or ('Training hinge loss' in line):
                    splits = line.split(',')
                    splits_tmp = splits[1]
                    sub_splits = splits_tmp.split(' ')
                    train_loss = np.float_(sub_splits[len(sub_splits)-1])
                if ('Test cross entropy loss' in line) or ('Test hinge loss' in line):
                    splits = line.split(' ')
                    g_epoch_id = np.int_(splits[1].split('.')[0])
                    splits = line.split(',')
                    splits_tmp = splits[1]
                    sub_splits = splits_tmp.split(' ')
                    test_loss = np.float_(sub_splits[len(sub_splits)-1])
                    if dataset == 'yfcc':
                        row = pd.Series([g_epoch_id, train_accuracy, test_accuracy, train_loss, test_loss], index=df_tmp_compute_error.columns)
                        df_tmp_compute_error.loc[len(df_tmp_compute_error)] = row
                        df_tmp_compute_error.reset_index()
                        if nr_procs == 256:
                            if scaling_type == 'weak':
                                scaling_type = 'strong'
                                row = pd.Series([g_epoch_id, train_accuracy, test_accuracy, train_loss, test_loss], index=df_tmp_compute_error.columns)
                                df_tmp_compute_error.loc[len(df_tmp_compute_error)] = row
                                df_tmp_compute_error.reset_index()
                                scaling_type = 'weak'
                else:
                    continue
                    
        df_merged = pd.merge(df_tmp, df_tmp_compute_error, on='g_epoch_id', how='inner')
        df_merged.reset_index()
        create_df = pd.concat([create_df, df_merged], ignore_index = True)
        file.close()


    create_df.to_csv(save_path, index = False)


def save_csv_UPMEM_benchmark(save_path, filenames_list, file_paths_list):
    columns = ['scaling_type', 'architecture', 'type_precision', 
           'algorithm', 'nr_procs', 'batch_size', 'lr', 'model_type',
           'reg_term', 'reg_term_alpha', 'num_global_epochs', 'num_local_epochs', 'start_up_time', 'total_start_up_time', 'total_communication_time', 'total_time', 'total_time_with_start_up_time',
           'CPU_DPU_init_band', 'CPU_DPU_init_time', 'CPU_DPU_init_data', 
           'comm_init', 'comm_per_epoch', 
           'CPU_and_DPU_band', 'CPU_and_DPU_time', 'CPU_and_DPU_data',
           'CPU_DPU_band', 'CPU_DPU_time', 'CPU_DPU_data',
           'DPU_CPU_band', 'DPU_CPU_time', 'DPU_CPU_data',
           'model_average_time',
           'DPU_init_time', 
           'DPU_compute_time',
           'M_and_W_band', 'M_and_W_time', 'M_and_W_data',
           'M_W_band', 'M_W_time', 'M_W_data',
           'W_M_band', 'W_M_time', 'W_M_data']
    create_df = pd.DataFrame(columns=columns)

    for k in range(0, len(filenames_list)):
        filename_path = filenames_list[k]
        filename = filenames_list[k].split('/')[-1]
        file_path = file_paths_list[k]
        architecture = 'DPU'
        dataset = ''
        if 'YFCC' in file_path:
            dataset = 'yfcc'
        elif 'Criteo' in file_path:
            dataset = 'criteo'

        scaling_type = ''
        if 'hyperparameter_search' in file_path:
            continue
        elif 'strong_scaling' in file_path:
            scaling_type = 'strong'
        else:
            scaling_type = 'weak'


        model_type = ''
        if 'lr_' in filename:
            model_type = 'lr'
        elif 'svm_' in filename:
            model_type = 'svm'
        else:
            print('Filename: {} has no model_type!!!\n'.format(filename))

        type_precision = 'uint32'

        
        algorithm = ''
        if 'admm' in file_path:
            algorithm = 'admm'
        elif 'mbsgd' in file_path:
            algorithm = 'mbsgd'
        elif 'ga_' in file_path:
            algorithm = 'ga'
        
        nr_procs = parse_filename('NR_DPUS', filename)
        batch_size = parse_filename('b_size_frac', filename)
        lr = parse_filename('learning_rate', filename)
        reg_term = parse_filename('reg_term', filename)
        if algorithm == 'admm':
            reg_term_alpha = parse_filename('alpha', filename)
        else:
            reg_term_alpha = np.nan
        num_global_epochs = parse_filename('epochs', filename)
        num_local_epochs = parse_filename_local_epochs(filename)
        if algorithm == 'ga' and num_local_epochs != 1:
            continue


        df_tmp = pd.DataFrame(columns=columns)


        file_tmp = open(file_path, 'r')
        start_successful = False  
        end_successful = False
        while (True):
            start_successful 
            line = file_tmp.readline()
            if not line:
                break
            if 'Experiment successfully started.' in line:
                start_successful = True
            if 'Experiment successfully completed.' in line:
                end_successful = True
        

        if (not start_successful) and (not end_successful):
            print('filename: {}\n'.format(filename))
            break
        file_tmp.close()

        file = open(file_path, 'r')

        start_up_time = 0.0
        total_start_up_time = 0.0
        total_communication_time = 0.0
        total_time = 0.0
        CPU_DPU_init_band = 0.0
        CPU_DPU_init_time = 0.0
        CPU_DPU_init_data = 0.0
        comm_init = 0
        comm_per_epoch = 0
        CPU_and_DPU_band = 0.0
        CPU_and_DPU_time = 0.0
        CPU_and_DPU_data = 0.0
        CPU_DPU_band = 0.0
        CPU_DPU_time = 0.0
        CPU_DPU_data = 0.0
        DPU_CPU_band = 0.0
        DPU_CPU_time = 0.0
        DPU_CPU_data = 0.0
        model_average_time = 0.0
        DPU_init_time = 0.0
        DPU_compute_time = 0.0
        M_and_W_band = 0.0
        M_and_W_time = 0.0
        M_and_W_data = 0.0
        M_W_band = 0.0
        M_W_time = 0.0
        M_W_data = 0.0
        W_M_band = 0.0
        W_M_time = 0.0
        W_M_data = 0.0

        while (True):
            line = file.readline()
            if not line:
                break
            if 'elapsed time for allocation of DPUs' in line:
                splits = line.split(' ')
                load_binary_to_DPU = np.float_(splits[len(splits)-2])
                start_up_time += load_binary_to_DPU
                total_start_up_time += load_binary_to_DPU
            if 'Preprocessing' in line:
                splits = line.split(' ')
                preprocessing = np.float_(splits[len(splits)-2])
                start_up_time += preprocessing
                total_start_up_time += preprocessing
            if 'Benchmark:CPU_DPU_initialization_bandwidth' in line:
                splits = line.split(',')
                sub_splits = splits[0].split(' ')
                CPU_DPU_init_band = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[1].split(' ')
                CPU_DPU_init_time = np.float_(sub_splits[len(sub_splits)-2])
                total_start_up_time += CPU_DPU_init_time
                sub_splits = splits[2].split(' ')
                CPU_DPU_init_data = np.float_(sub_splits[len(sub_splits)-2])
            if 'Benchmark:communications_init' in line:
                splits = line.split(',')
                sub_splits = splits[0].split(' ')
                comm_init = np.float_(sub_splits[len(sub_splits)-1])
                sub_splits = splits[1].split(' ')
                comm_per_epoch = np.float_(sub_splits[len(sub_splits)-1])
            if 'Benchmark:CPU_and_DPU_combined_epoch_bandwidth' in line:
                splits = line.split(',')
                sub_splits = splits[0].split(' ')
                CPU_and_DPU_band = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[1].split(' ')
                CPU_and_DPU_time = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[2].split(' ')
                CPU_and_DPU_data = np.float_(sub_splits[len(sub_splits)-2])
            if 'Benchmark:CPU_DPU_epoch_bandwidth' in line:
                splits = line.split(',')
                sub_splits = splits[0].split(' ')
                CPU_DPU_band = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[1].split(' ')
                CPU_DPU_time = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[2].split(' ')
                CPU_DPU_data = np.float_(sub_splits[len(sub_splits)-2])
            if 'Benchmark:DPU_CPU_epoch_bandwidth' in line:
                splits = line.split(',')
                sub_splits = splits[0].split(' ')
                DPU_CPU_band = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[1].split(' ')
                DPU_CPU_time = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[2].split(' ')
                DPU_CPU_data = np.float_(sub_splits[len(sub_splits)-2])
            if 'Benchmark:epoch_time_model_average' in line:
                splits = line.split(',')
                sub_splits = splits[0].split(' ')
                model_average_time = np.float_(sub_splits[len(sub_splits)-2])
            if 'Benchmark:DPU_epoch_initialization_time' in line:
                splits = line.split(' ')
                DPU_init_time = np.float_(splits[len(splits)-2])
            if 'Benchmark:DPU_epoch_compute_time' in line:
                splits = line.split(' ')
                DPU_compute_time = np.float_(splits[len(splits)-2])
            if 'Benchmark:mram_and_wram_combined_epoch_bandwidth' in line:
                splits = line.split(',')
                sub_splits = splits[0].split(' ')
                M_and_W_band = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[1].split(' ')
                M_and_W_time = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[2].split(' ')
                M_and_W_data = np.float_(sub_splits[len(sub_splits)-2])
            if 'Benchmark:mram_to_wram_epoch_bandwidth' in line:
                splits = line.split(',')
                sub_splits = splits[0].split(' ')
                M_W_band = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[1].split(' ')
                M_W_time = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[2].split(' ')
                M_W_data = np.float_(sub_splits[len(sub_splits)-2])
            if 'Benchmark:wram_to_mram_epoch_bandwidth' in line:
                splits = line.split(',')
                sub_splits = splits[0].split(' ')
                W_M_band = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[1].split(' ')
                W_M_time = np.float_(sub_splits[len(sub_splits)-2])
                sub_splits = splits[2].split(' ')
                W_M_data = np.float_(sub_splits[len(sub_splits)-2])
                
                total_communication_time = model_average_time + CPU_and_DPU_time
                total_time_with_start_up_time = total_start_up_time + total_communication_time + DPU_compute_time + M_and_W_time
                total_time = total_communication_time + DPU_compute_time + M_and_W_time
                row = pd.Series([scaling_type, architecture, type_precision, 
                        algorithm, nr_procs, batch_size, lr, model_type,
                        reg_term, reg_term_alpha, num_global_epochs, num_local_epochs, start_up_time, total_start_up_time, total_communication_time, total_time, total_time_with_start_up_time,
                        CPU_DPU_init_band, CPU_DPU_init_time, CPU_DPU_init_data, 
                        comm_init, comm_per_epoch, 
                        CPU_and_DPU_band, CPU_and_DPU_time, CPU_and_DPU_data,
                        CPU_DPU_band, CPU_DPU_time, CPU_DPU_data,
                        DPU_CPU_band, DPU_CPU_time, DPU_CPU_data,
                        model_average_time,
                        DPU_init_time, 
                        DPU_compute_time,
                        M_and_W_band, M_and_W_time, M_and_W_data,
                        M_W_band, M_W_time, M_W_data,
                        W_M_band, W_M_time, W_M_data], index=df_tmp.columns)
                df_tmp.loc[len(df_tmp)] = row
                df_tmp.reset_index()
                if nr_procs == 256:
                    if scaling_type == 'weak':
                        scaling_type = 'strong'
                        row = pd.Series([scaling_type, architecture, type_precision, 
                        algorithm, nr_procs, batch_size, lr, model_type,
                        reg_term, reg_term_alpha, num_global_epochs, num_local_epochs, start_up_time, total_start_up_time, total_communication_time, total_time, total_time_with_start_up_time,
                        CPU_DPU_init_band, CPU_DPU_init_time, CPU_DPU_init_data, 
                        comm_init, comm_per_epoch, 
                        CPU_and_DPU_band, CPU_and_DPU_time, CPU_and_DPU_data,
                        CPU_DPU_band, CPU_DPU_time, CPU_DPU_data,
                        DPU_CPU_band, DPU_CPU_time, DPU_CPU_data,
                        model_average_time,
                        DPU_init_time, 
                        DPU_compute_time,
                        M_and_W_band, M_and_W_time, M_and_W_data,
                        M_W_band, M_W_time, M_W_data,
                        W_M_band, W_M_time, W_M_data], index=df_tmp.columns)
                        df_tmp.loc[len(df_tmp)] = row
                        df_tmp.reset_index()
                        scaling_type = 'weak'

        df_tmp.reset_index()
        create_df = pd.concat([create_df, df_tmp], ignore_index = True)
        file.close()

    create_df.to_csv(save_path, index = False)

    
