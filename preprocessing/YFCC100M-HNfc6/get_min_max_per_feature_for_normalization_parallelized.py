import os
import sys
import numpy as np
import pandas as pd
import re
import time

from multiprocessing import Process, shared_memory
import ctypes  

def find_filenames(root_folder, tag):
    filepaths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if tag in file:
                filepaths.append(os.path.join(root_folder, file))

    return filepaths

def extract_filename_number(filename):
    number = -1
    match = re.search(r'fc6_(\d+)_tag', filename)
    if match:
        number = int(match.group(1))
    return number

def process_function_old(process_id, nr_jobs, filepath, shm_min_value, shared_array_min_values, shm_max_values, shared_array_max_values):
    dtype = np.float32
    exis
    print("Start job_id = {}".format(process_id))
    print("job_id = {} processes file: {}".format(job_id, filepath))
    with open(filepath, 'r') as file:
        count = 0
        while(True):
            line = file.readline()
            if not line:
                break
            splits = line.split()
            if len(splits) == 4097:
                splits_float = [float(x) for x in splits[1:4097]]
                for i in range(0,4096):
                    if (splits_float[i] < shm_min_value[nr_jobs*i + process_id]):
                        shm_min_value[nr_jobs*i + process_id] = splits_float[i]
                    if (splits_float[i] > shm_max_value[nr_jobs*i + process_id]):
                        shm_min_value[nr_jobs*i + process_id] = splits_float[i]
            count += 1
    print("Done job_id = {}".format(process_id))

def process_function(job_id, nr_jobs, filepath, shm_min_name, shm_max_name, shape, dtype):
    existing_shm_min = shared_memory.SharedMemory(name=shm_min_name)
    shared_array_min_values = np.ndarray(shape, dtype=dtype, buffer=existing_shm_min.buf)
    
    existing_shm_max = shared_memory.SharedMemory(name=shm_max_name)
    shared_array_max_values = np.ndarray(shape, dtype=dtype, buffer=existing_shm_max.buf)

    print("Start job_id = {}".format(job_id))
    print("job_id = {} processes file: {}".format(job_id, filepath))
    with open(filepath, 'r') as file:
        count = 0
        while True:
            line = file.readline()
            if not line:
                break
            splits = line.split()
            if len(splits) == 4097:
                splits_float = [float(x) for x in splits[1:4097]]
                for i in range(4096):
                    if splits_float[i] < shared_array_min_values[nr_jobs*i + job_id]:
                        shared_array_min_values[nr_jobs*i + job_id] = splits_float[i]
                    if splits_float[i] > shared_array_max_values[nr_jobs*i + job_id]:
                        shared_array_max_values[nr_jobs*i + job_id] = splits_float[i]
            count += 1
    print("Done job_id = {}".format(job_id))

    existing_shm_min.close()
    existing_shm_max.close()


if __name__ == "__main__":
    source_path_tmp = str(sys.argv[1])
    time.sleep(5)
    print("Start processing file 0") 


    filepaths = find_filenames(source_path_tmp + "preprocessing/YFCC100M-HNfc6/initial_preprocessing", "indoor")
    min_values_indoor = np.zeros(97*4096, dtype=np.float32)
    max_values_indoor = np.zeros(97*4096, dtype=np.float32)
    shm_min_values_indoor = shared_memory.SharedMemory(create=True, size=min_values_indoor.nbytes)
    shared_array_min_values_indoor = np.ndarray(min_values_indoor.shape, dtype=min_values_indoor.dtype, buffer=shm_min_values_indoor.buf)
    np.copyto(shared_array_min_values_indoor, min_values_indoor)
    shm_max_values_indoor = shared_memory.SharedMemory(create=True, size=max_values_indoor.nbytes)
    shared_array_max_values_indoor = np.ndarray(max_values_indoor.shape, dtype=max_values_indoor.dtype, buffer=shm_max_values_indoor.buf)
    np.copyto(shared_array_max_values_indoor, max_values_indoor)




    processes = []
    nr_jobs = len(filepaths)
    shape = min_values_indoor.shape
    dtype = min_values_indoor.dtype
    for k, filename in enumerate(filepaths):
        job_id = extract_filename_number(filename)
        process = Process(target=process_function, args=(job_id, nr_jobs, filename, shm_min_values_indoor.name, shm_max_values_indoor.name, shape, dtype))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()

    min_values_indoor = []
    max_values_indoor = []

    for i in range(0,4096):
        min_values_indoor.append(int(0))
        max_values_indoor.append(int(0))
    
    for i in range(0, 4096):
        min_values_indoor[i] = np.min(shared_array_min_values_indoor[(i*nr_jobs):((i+1)*nr_jobs)])
        max_values_indoor[i] = np.max(shared_array_max_values_indoor[(i*nr_jobs):((i+1)*nr_jobs)])

    time.sleep(5)
    print("Start processing file 1") 


    filepaths = find_filenames(source_path_tmp + "preprocessing/YFCC100M-HNfc6/initial_preprocessing", "outdoor")
    min_values_outdoor = np.zeros(97*4096, dtype=np.float32)
    max_values_outdoor = np.zeros(97*4096, dtype=np.float32)
    shm_min_values_outdoor = shared_memory.SharedMemory(create=True, size=min_values_outdoor.nbytes)
    shared_array_min_values_outdoor = np.ndarray(min_values_outdoor.shape, dtype=min_values_outdoor.dtype, buffer=shm_min_values_outdoor.buf)
    np.copyto(shared_array_min_values_outdoor, min_values_outdoor)
    shm_max_values_outdoor = shared_memory.SharedMemory(create=True, size=max_values_outdoor.nbytes)
    shared_array_max_values_outdoor = np.ndarray(max_values_outdoor.shape, dtype=max_values_outdoor.dtype, buffer=shm_max_values_outdoor.buf)
    np.copyto(shared_array_max_values_outdoor, max_values_outdoor)




    processes = []
    nr_jobs = len(filepaths)
    shape = min_values_outdoor.shape
    dtype = min_values_outdoor.dtype
    for k, filename in enumerate(filepaths):
        job_id = extract_filename_number(filename)
        process = Process(target=process_function, args=(job_id, nr_jobs, filename, shm_min_values_outdoor.name, shm_max_values_outdoor.name, shape, dtype))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()



    

    min_values_outdoor = []
    max_values_outdoor = []

    for i in range(0,4096):
        min_values_outdoor.append(int(0))
        max_values_outdoor.append(int(0))

    for i in range(0, 4096):
        min_values_outdoor[i] = np.min(shared_array_min_values_outdoor[(i*nr_jobs):((i+1)*nr_jobs)])
        max_values_outdoor[i] = np.max(shared_array_max_values_outdoor[(i*nr_jobs):((i+1)*nr_jobs)])

    min_values = []
    max_values = []

    for i in range(0,4096):
        min_values.append(int(0))
        max_values.append(int(0))

    for i in range(0, 4096):
        min_values[i] = np.min([min_values_indoor[i], min_values_outdoor[i]])
        max_values[i] = np.max([max_values_indoor[i], max_values_outdoor[i]])
    
    

    dst_file = open(source_path_tmp + 'preprocessing/YFCC100M-HNfc6/min_max_per_feature_for_normalization_parallelized.txt', 'w+')
    dst_file.write('min_values\n')
    for i in range(0,4096):
        if (i != 4095):
            dst_file.write('{:.3f},'.format(min_values[i]))
        else:
            dst_file.write('{:.3f}'.format(min_values[i]))

    dst_file.write('\n')
    dst_file.write('max_values\n')
    for i in range(0,4096):
        if (i != 4095):
            dst_file.write('{:.3f},'.format(max_values[i]))
        else:
            dst_file.write('{:.3f}'.format(max_values[i]))

    dst_file.close()

    shm_min_values_outdoor.close()
    shm_min_values_outdoor.unlink()
    shm_max_values_outdoor.close()
    shm_max_values_outdoor.unlink()
    print("Done get_min_max_per_feature_for_normalization.py")

    
    

