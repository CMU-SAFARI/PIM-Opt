import os, sys
import numpy as np
import pandas as pd

from multiprocessing import Process, shared_memory
import ctypes


from sklearn.metrics import roc_auc_score

from utils_postprocessing_results_UPMEM_Criteo import *


def sub_process_function(sub_process_id, process_id, model_type, NR_DPUS, dest_path, m_size, m_size_test, n_size, W, X, Y, X_test, Y_test, filename, file_path, shm_auc_roc_train_name, shared_array_auc_roc_train_shape, shm_auc_roc_test_name, shared_array_auc_roc_test_shape, shm_test_loss_name, shared_array_test_loss_shape):
    dtype = np.float32
    existing_shm_auc_roc_train = shared_memory.SharedMemory(name=shm_auc_roc_train_name)
    shared_array_auc_roc_train = np.ndarray(shared_array_auc_roc_train_shape, dtype=dtype, buffer=existing_shm_auc_roc_train.buf)
    existing_shm_auc_roc_test = shared_memory.SharedMemory(name=shm_auc_roc_test_name)
    shared_array_auc_roc_test = np.ndarray(shared_array_auc_roc_test_shape, dtype=dtype, buffer=existing_shm_auc_roc_test.buf)
    existing_shm_test_loss = shared_memory.SharedMemory(name=shm_test_loss_name)
    shared_array_test_loss = np.ndarray(shared_array_test_loss_shape, dtype=dtype, buffer=existing_shm_test_loss.buf)

    print("NR_DPUS = {}. Process_id = {}. Start: {}\n".format(NR_DPUS, process_id, filename))
    if model_type == 'lr':
        print("NR_DPUS = {}. Process_id = {}. Start sub_process_id {}\n".format(NR_DPUS, process_id, sub_process_id))
        Y_train_probs = log_reg_compute_probs(m_size,X,W,sub_process_id)
        print("NR_DPUS = {}. Process_id = {}. Done computing Y_train_probs\n".format(NR_DPUS, process_id))
        Y_test_probs = log_reg_compute_probs(m_size_test,X_test,W,sub_process_id)
        print("NR_DPUS = {}. Process_id = {}. Done computing Y_test_probs\n".format(NR_DPUS, process_id))
        auc_roc_train_current = roc_auc_score(Y, Y_train_probs)
        print("NR_DPUS = {}. Process_id = {}. sub_process_id = {}. auc_roc_train_current = {}\n".format(NR_DPUS, process_id, sub_process_id, auc_roc_train_current))
        auc_roc_test_current = roc_auc_score(Y_test, Y_test_probs)
        print("NR_DPUS = {}. Process_id = {}. sub_process_id = {}. auc_roc_test_current = {}\n".format(NR_DPUS, process_id, sub_process_id, auc_roc_test_current))
        shared_array_auc_roc_train[process_id*10 + sub_process_id] = np.float32(auc_roc_train_current)
        shared_array_auc_roc_test[process_id*10 + sub_process_id] = np.float32(auc_roc_test_current)
        test_loss_current = np.float32(0) 
        shared_array_test_loss[process_id*10 + sub_process_id] = np.float32(test_loss_current)

        print("NR_DPUS = {}. Process_id = {}. End sub_process_id {}\n".format(NR_DPUS, process_id, sub_process_id))

    else:
        print("NR_DPUS = {}. Process_id = {}. Start sub_process_id {}\n".format(NR_DPUS, process_id, sub_process_id))
        Y_train_decision_values = svm_compute_decision_values(m_size,X,W,sub_process_id)
        print("NR_DPUS = {}. Process_id = {}. Done computing Y_train_decision_values\n".format(NR_DPUS, process_id))
        Y_test_decision_values = svm_compute_decision_values(m_size_test,X_test,W,sub_process_id)
        print("NR_DPUS = {}. Process_id = {}. Done computing Y_test_probs\n".format(NR_DPUS, process_id))
        auc_roc_train_current = roc_auc_score(Y, Y_train_decision_values)
        print("NR_DPUS = {}. Process_id = {}. sub_process_id = {}. auc_roc_train_current = {}\n".format(NR_DPUS, process_id, sub_process_id, auc_roc_train_current))
        auc_roc_test_current = roc_auc_score(Y_test, Y_test_decision_values)
        print("NR_DPUS = {}. Process_id = {}. sub_process_id = {}. auc_roc_test_current = {}\n".format(NR_DPUS, process_id, sub_process_id, auc_roc_test_current))
        shared_array_auc_roc_train[process_id*10 + sub_process_id] = np.float32(auc_roc_train_current)
        shared_array_auc_roc_test[process_id*10 + sub_process_id] = np.float32(auc_roc_test_current)
        

        test_loss_current = np.float32(0) 
        shared_array_test_loss[process_id*10 + sub_process_id] = np.float32(test_loss_current)

        
        print("NR_DPUS = {}. Process_id = {}. End sub_process_id {}\n".format(NR_DPUS, process_id, sub_process_id))


    existing_shm_auc_roc_train.close()
    existing_shm_auc_roc_test.close()
    existing_shm_test_loss.close()


def process_function(process_id, NR_DPUS,dest_path, m_size, m_size_test, n_size, X, Y, X_test, Y_test, filename, file_path, shm_auc_roc_train_name, shared_array_auc_roc_train_shape, shm_auc_roc_test_name, shared_array_auc_roc_test_shape, shm_test_loss_name, shared_array_test_loss_shape):
    dtype = np.float32
    existing_shm_auc_roc_train = shared_memory.SharedMemory(name=shm_auc_roc_train_name)
    shared_array_auc_roc_train = np.ndarray(shared_array_auc_roc_train_shape, dtype=dtype, buffer=existing_shm_auc_roc_train.buf)
    existing_shm_auc_roc_test = shared_memory.SharedMemory(name=shm_auc_roc_test_name)
    shared_array_auc_roc_test = np.ndarray(shared_array_auc_roc_test_shape, dtype=dtype, buffer=existing_shm_auc_roc_test.buf)
    existing_shm_test_loss = shared_memory.SharedMemory(name=shm_test_loss_name)
    shared_array_test_loss = np.ndarray(shared_array_test_loss_shape, dtype=dtype, buffer=existing_shm_test_loss.buf)
    model_type = ''
    if 'lr_' in filename:
        model_type = 'lr'
    else:
        model_type = 'svm'
    
    W = get_models_W_from_file(file_path, n_size)
    

    processes = []
    for i in range(0,10): 
        process = Process(target=sub_process_function, args=(i, process_id, model_type, NR_DPUS, dest_path, m_size, m_size_test, n_size, W, X, Y, X_test, Y_test, filename, file_path, shm_auc_roc_train_name, shared_array_auc_roc_train_shape, shm_auc_roc_test_name, shared_array_auc_roc_test_shape, shm_test_loss_name, shared_array_test_loss_shape))
        processes.append(process)
        process.start()

        
    for process in processes:
        process.join()



    dest_filename = dest_path + '/' + filename
    write_auc_roc_train = np.zeros(10, dtype=np.float32)
    write_auc_roc_test = np.zeros(10, dtype=np.float32)
    write_test_loss = np.zeros(10, dtype=np.float32)
    for i in range(0,10):
        write_auc_roc_train[i] = np.float32(shared_array_auc_roc_train[process_id*10 + i])
        write_auc_roc_test[i] = np.float32(shared_array_auc_roc_test[process_id*10 + i])
        write_test_loss[i] = np.float32(shared_array_test_loss[process_id*10 + i])


    write_results_to_file_just_for_fixing_test_loss(file_path, dest_filename, write_auc_roc_train, write_auc_roc_test, write_test_loss, model_type)
    print("NR_DPUS = {}. Process_id = {}. End: {}\n".format(NR_DPUS, process_id, filename))
    
    existing_shm_auc_roc_train.close()
    existing_shm_auc_roc_test.close()
    existing_shm_test_loss.close()





def main():
    NR_DPUS = int(sys.argv[1])
    source_path = str(sys.argv[2])
    dest_path = str(sys.argv[3])
    source_path_root = str(sys.argv[4])
    
    

    strong_scaling = 0
    if "strong" in source_path:
        strong_scaling = 1
    print("Start NR_DPUS = {}.\n".format(NR_DPUS))
    
    filenames_list, filepaths_list = find_txt_filenames_new(source_path)

    auc_roc_train = np.zeros(120, dtype=np.float32)
    auc_roc_test = np.zeros(120, dtype=np.float32)
    test_loss = np.zeros(120, dtype=np.float32)
    shm_auc_roc_train = shared_memory.SharedMemory(create=True, size=auc_roc_train.nbytes)
    shared_array_auc_roc_train = np.ndarray(auc_roc_train.shape, dtype=auc_roc_train.dtype, buffer=shm_auc_roc_train.buf)
    np.copyto(shared_array_auc_roc_train, auc_roc_train)
    shm_auc_roc_test = shared_memory.SharedMemory(create=True, size=auc_roc_test.nbytes)
    shared_array_auc_roc_test = np.ndarray(auc_roc_test.shape, dtype=auc_roc_test.dtype, buffer=shm_auc_roc_test.buf)
    np.copyto(shared_array_auc_roc_test, auc_roc_test)
    shm_test_loss = shared_memory.SharedMemory(create=True, size=test_loss.nbytes)
    shared_array_test_loss = np.ndarray(test_loss.shape, dtype=test_loss.dtype, buffer=shm_test_loss.buf)
    np.copyto(shared_array_test_loss, test_loss)

    m_size = 0
    m_size_test = 178236537

    n_size = 1000000

    if (NR_DPUS == 256 or strong_scaling == 1):
        m_size = 50331648
    elif (NR_DPUS == 512):
        m_size = 100663296
    elif (NR_DPUS == 1024):
        m_size = 201326592
    elif (NR_DPUS == 2048):
        m_size = 402653184
    

    X_test, Y_test = read_test_data(m_size_test, source_path_root)
    print("NR_DPUS = {}. Done reading in test data.\n".format(NR_DPUS))
    X, Y = read_train_data(NR_DPUS, strong_scaling, m_size, source_path_root)
    print("NR_DPUS = {}. Done reading in train data.\n".format(NR_DPUS))
    
    filenames_list_threads = []
    filepaths_list_threads = []

    for k in range(0,len(filenames_list)):
        filename = filenames_list[k]
        file_path = filepaths_list[k]
        if (NR_DPUS != parse_filename('NR_DPUS', filename)):
            continue
        filenames_list_threads.append(filename)
        filepaths_list_threads.append(file_path)
        print(file_path)
    
    

    processes = []
    for i in range(0, len(filenames_list_threads)): 
        process = Process(target=process_function, args=(i, NR_DPUS, dest_path, m_size, m_size_test, n_size, X, Y, X_test, Y_test, filenames_list_threads[i], filepaths_list_threads[i],shm_auc_roc_train.name, shared_array_auc_roc_train.shape, shm_auc_roc_test.name, shared_array_auc_roc_test.shape, shm_test_loss.name, shared_array_test_loss.shape))
        processes.append(process)
        process.start()
        
    for process in processes:
        process.join()
    
    shm_auc_roc_train.close()
    shm_auc_roc_test.close()
    shm_test_loss.close()
    shm_auc_roc_train.unlink()
    shm_auc_roc_test.unlink()
    shm_test_loss.unlink()


if __name__ == "__main__":
    main()