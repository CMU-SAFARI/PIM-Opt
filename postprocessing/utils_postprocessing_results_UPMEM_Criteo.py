import os, sys
import numpy as np
import pandas as pd





def find_txt_filenames_new(root_folder):
    filenames_list = []
    filepaths_list = []
    for dirpath, dirnames, files in os.walk(root_folder):
        if dirpath.split(os.sep)[-1].startswith(("admm", "mbsgd","ga")):
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


def read_train_data(NR_DPUS, strong_scaling, m_size, source_path_root):
    X = np.zeros((m_size, 39), dtype=np.uint32)
    Y = np.zeros(m_size, dtype=np.uint32)
    filename = ''
    counter = 0
    if (NR_DPUS == 256 or strong_scaling == 1):
        filename = source_path_root + 'preprocessed_datasets/Criteo/Criteo_train_NR_DPUS_256_label_0_48610305_label_1_1721343_total_50331648.txt'
    elif (NR_DPUS == 512):
        filename = source_path_root + 'preprocessed_datasets/Criteo/Criteo_train_NR_DPUS_512_label_0_97220610_label_1_3442686_total_100663296.txt'
    elif (NR_DPUS == 1024):
        filename = source_path_root + 'preprocessed_datasets/Criteo/Criteo_train_NR_DPUS_1024_label_0_194441220_label_1_6885372_total_201326592.txt'
    elif (NR_DPUS == 2048):
        filename = source_path_root + 'preprocessed_datasets/Criteo/Criteo_train_NR_DPUS_2048_label_0_388882440_label_1_13770744_total_402653184.txt'
    file_handle = open(filename, 'r')

    while(counter < m_size):
        line = file_handle.readline()
        splits = line.split()
        
        label = np.uint32(splits[0])
        splits_uint32 = [np.uint32(z) for z in splits[1:40]]
        Y[counter] = label
        for i in range(0,39):
            X[counter][i] = splits_uint32[i]

        counter += 1

    
    file_handle.close()
    return X, Y


def read_test_data(m_size_test, source_path_root):
    X_test = np.zeros((m_size_test, 39), dtype=np.uint32)
    Y_test = np.zeros(m_size_test, dtype=np.uint32)
    filename = source_path_root + 'preprocessed_datasets/Criteo/test_data_criteo_tb_day_23.txt'
    counter = 0
    
    file_handle = open(filename, 'r')

    while(counter < m_size_test):
        line = file_handle.readline()
        splits = line.split()
        
        label = np.uint32(splits[0])
        splits_uint32 = [np.uint32(z) for z in splits[1:40]]
        Y_test[counter] = label
        for i in range(0,39):
            X_test[counter][i] = splits_uint32[i]

        counter += 1

    
    file_handle.close()
    return X_test, Y_test

def get_models_W_from_file(file_path, n_size):
    W = np.zeros((10, n_size), dtype=np.float32)
    file_handle = open(file_path, 'r')
    counter = 0
    search_string = ''
    splits = file_path.split('/')
    if 'ADMM' in splits[len(splits)-1]:
        search_string = '. Model W_ADMM_global.'
    else:
        search_string = '. Model W_dpu_fp.'

    while (True):
        line = file_handle.readline()
        if not line:
            break
        if search_string in line:
            line = file_handle.readline()
            splits = line.split(',')
            splits_float = [np.float32(z) for z in splits]
            for i in range(0, n_size):
                W[counter][i] = splits_float[i]

            counter += 1
    return W

def sort_Y_according_to_indices(m_size, sorted_Y_decision_values, Y_copy):
    sorted_Y_copy = np.zeros(m_size, dtype=np.float32)
    for i in range(0, m_size):

        sorted_Y_copy = Y_copy[sorted_Y_decision_values]
    return sorted_Y_copy


def log_reg_compute_cross_entropy_loss(m_size,Y,Y_probs):
    cross_entropy_loss = np.float64(0)
    for i in range(0, m_size):
        sigmoid_temp = np.float64(Y_probs[i])
        cross_entropy_loss_tmp = np.float64(0)
        if Y[i] == 1:
            cross_entropy_loss_tmp = np.float64(np.log(sigmoid_temp))
        else:
            cross_entropy_loss_tmp = np.float64(np.log(np.float64(1)-sigmoid_temp))
        if Y[i] == 0:
            cross_entropy_loss_tmp *= np.float64(0.5)
        else:
            cross_entropy_loss_tmp *= np.float64(15)
        cross_entropy_loss += cross_entropy_loss_tmp
    cross_entropy_loss = cross_entropy_loss/np.float64(m_size)
    cross_entropy_loss = -cross_entropy_loss
    return cross_entropy_loss
        
    



def log_reg_compute_probs(m_size,X,W,current_epoch):
    Y_probs = np.zeros(m_size, dtype=np.float32)
    for i in range(0, m_size):
        dot_product = np.float64(0)
        tmp = np.uint32(0)
        dot_product += np.float64(W[current_epoch][0]) 
        for j in range(0,39):
            tmp = X[i][j]
            dot_product += np.float64(W[current_epoch][tmp])
        sigmoid_temp = np.float32(np.float64(1)/(np.float64(1)+np.exp(-dot_product)))
        Y_probs[i] = np.float32(sigmoid_temp) 
    return Y_probs

def svm_compute_hinge_loss(m_size,Y,Y_decision_values):
    hinge_loss = np.float64(0)
    for i in range(0, m_size):
        dot_product = np.float64(Y_decision_values[i])
        hinge_loss_tmp = np.float64(0)
        if Y[i] == 1:
            hinge_loss_tmp = np.float64(1.0)
        else:
            hinge_loss_tmp = np.float64(-1.0)
        hinge_loss_tmp = np.float64(0.5) - hinge_loss_tmp * dot_product

        if Y[i] == 0:
            hinge_loss_tmp *= np.float64(0.5)
        else:
            hinge_loss_tmp *= np.float64(15)
        if hinge_loss_tmp >= 0:
            hinge_loss += hinge_loss_tmp   
    hinge_loss = hinge_loss / np.float64(m_size)     
    return hinge_loss


def svm_compute_decision_values(m_size,X,W,current_epoch):
    Y_decision_values = np.zeros(m_size, dtype=np.float32)
    for i in range(0, m_size):
        dot_product = np.float64(0)
        tmp = np.uint32(0)
        dot_product += np.float64(W[current_epoch][0]) # add bias
        for j in range(0,39):
            tmp = X[i][j]
            dot_product += np.float64(W[current_epoch][tmp])
        Y_decision_values[i] = np.float32(dot_product)
    return Y_decision_values
        

def write_results_to_file(file_path, dest_filename, write_auc_roc_train, write_auc_roc_test, model_type):
    source_file_handle = open(file_path, 'r')
    dest_file_handle = open(dest_filename, 'w+')
    counter = 0
    search_string_train = ''
    search_string_test = ''
    if model_type == 'lr':
        search_string_train = 'Training cross entropy loss of averaged model'
        search_string_test = 'Test cross entropy loss of averaged model'
    else:
        search_string_train = 'Training hinge loss of averaged model'
        search_string_test = 'Test hinge loss of averaged model'

    while (True):
        line = source_file_handle.readline()
        if not line:
            break
        if search_string_train in line:
            dest_file_handle.write(line)
            dest_file_handle.write('Epoch {}. Train. roc_auc_score={}\n'.format(counter, float(write_auc_roc_train[counter])))
        elif search_string_test in line:
            dest_file_handle.write(line)
            dest_file_handle.write('Epoch {}. Test. roc_auc_score={}\n'.format(counter, float(write_auc_roc_test[counter])))
            counter += 1
        else:
            dest_file_handle.write(line)
    source_file_handle.close()
    dest_file_handle.close()


def write_results_to_file_just_for_fixing_test_loss(file_path, dest_filename, write_auc_roc_train, write_auc_roc_test, write_test_loss, model_type):
    source_file_handle = open(file_path, 'r')
    dest_file_handle = open(dest_filename, 'w+')
    algorithm = ''
    splits = file_path.split('/')
    if 'ADMM' in splits[len(splits)-1]:
        algorithm = 'ADMM'
    else:
        algorithm = 'MB_SGD'
    counter = 0
    search_string_train = ''
    search_string_test = ''
    if model_type == 'lr':
        search_string_train = 'Training cross entropy loss of averaged model'
        search_string_test = 'Test cross entropy loss of averaged model'
    else:
        search_string_train = 'Training hinge loss of averaged model'
        search_string_test = 'Test hinge loss of averaged model'

    while (True):
        line = source_file_handle.readline()
        if not line:
            break
        if search_string_train in line:
            dest_file_handle.write(line)
            dest_file_handle.write('Epoch {}. Train. roc_auc_score={}\n'.format(counter, float(write_auc_roc_train[counter])))
        elif search_string_test in line:
            dest_file_handle.write(line)
            dest_file_handle.write('Epoch {}. Test. roc_auc_score={}\n'.format(counter, float(write_auc_roc_test[counter])))
            counter += 1
        else:
            dest_file_handle.write(line)
    source_file_handle.close()
    dest_file_handle.close()



    


