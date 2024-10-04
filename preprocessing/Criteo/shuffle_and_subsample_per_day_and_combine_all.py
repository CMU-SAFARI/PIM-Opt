import os
import sys
import numpy as np
import random 
from itertools import islice



def grab_all_combined_statistics(dest_path_prefix):
    number_of_samples_label_0_per_day = []
    number_of_samples_label_1_per_day = []
    for job_id in range(0,23):
        
        

        current_file = open(dest_path_prefix + "preprocessing/Criteo/statistics/statistics_parallel_job_id_{}.txt".format(job_id), "r")
        while(True):
            line = current_file.readline()
            if not line:
                break
            
            splits = line.split()
            if "# label_0 =" in line:
                number_of_samples_label_0_per_day.append(splits[len(splits)-1])
            elif "# label_1 =" in line:
                number_of_samples_label_1_per_day.append(splits[len(splits)-1])
            elif "Overall: Categorical features" in line:
                break
        current_file.close()
    print("{},{}".format(len(number_of_samples_label_0_per_day), len(number_of_samples_label_1_per_day)))
    return number_of_samples_label_0_per_day, number_of_samples_label_1_per_day







if __name__ == "__main__":
    NR_DPUS = int(sys.argv[1])
    dest_path_prefix = str(sys.argv[2])

    total_number_label_0 = 0
    total_number_label_1 = 0
    total_number = 0
    number_label_0_per_day_per_DPU = 0
    number_label_1_per_day_per_DPU = 0

    if NR_DPUS == 256:
        total_number_label_0 = np.int64(48610305)
        total_number_label_1 = np.int64(1721343)
    if NR_DPUS == 512:
        total_number_label_0 = np.int64(97220610)
        total_number_label_1 = np.int64(3442686)
    if NR_DPUS == 1024:
        total_number_label_0 = np.int64(194441220)
        total_number_label_1 = np.int64(6885372)
    if NR_DPUS == 2048:
        total_number_label_0 = np.int64(388882440)
        total_number_label_1 = np.int64(13770744)
    total_number = total_number_label_0 + total_number_label_1
    
    number_label_0_per_day_per_DPU = np.int64(np.ceil(total_number_label_0/23))
    number_label_1_per_day_per_DPU = np.int64(np.ceil(total_number_label_1/23))

    number_of_samples_label_0_per_day = []
    number_of_samples_label_1_per_day = []

    number_of_samples_label_0_per_day, number_of_samples_label_1_per_day = grab_all_combined_statistics(dest_path_prefix)

    seed = 41

    train_data = []

    print("{},{}".format(len(number_of_samples_label_0_per_day), len(number_of_samples_label_1_per_day)))
    for job_id in range(0,23):
        print("job_id_{}: Start processing\n".format(job_id))
        current_file_label_0 = dest_path_prefix + "preprocessing/Criteo/preprocessed_label_0_label_1/criteo_tb_label_0_job_id_{}.txt".format(job_id)
        seed += 1 # always update seed before it is used
        np.random.seed(seed)
        train_data_label_0_current = []

        index = np.int64(number_of_samples_label_0_per_day[job_id])
        sample_ind_array_label_0_current = np.random.choice(index, number_label_0_per_day_per_DPU, replace=False)
        sample_ind_label_0_current = set(sample_ind_array_label_0_current)

        with open(current_file_label_0, "r") as infile:
            for current_index, line in enumerate(infile):
                if not sample_ind_label_0_current:
                    break
                if current_index in sample_ind_label_0_current:
                    train_data_label_0_current.append(line)
                    sample_ind_label_0_current.remove(current_index)  
        

        current_file_label_1 = dest_path_prefix + "preprocessing/Criteo/preprocessed_label_0_label_1/criteo_tb_label_1_job_id_{}.txt".format(job_id)
        seed += 1 # always update seed before it is used
        np.random.seed(seed)
        train_data_label_1_current = []

        index = np.int64(number_of_samples_label_1_per_day[job_id])
        sample_ind_array_label_1_current = np.random.choice(index, number_label_1_per_day_per_DPU, replace=False)
        sample_ind_label_1_current = set(sample_ind_array_label_1_current)

        with open(current_file_label_1, "r") as infile:
            for current_index, line in enumerate(infile):
                if not sample_ind_label_1_current:
                    break
                if current_index in sample_ind_label_1_current:
                    train_data_label_1_current.append(line)
                    sample_ind_label_1_current.remove(current_index)  
        
        train_data_combined_current = train_data_label_0_current + train_data_label_1_current
        seed += 1
        np.random.seed(seed)
        random.shuffle(train_data_combined_current)

        train_data.extend(train_data_combined_current)

        print("job_id_{}: End processing\n".format(job_id))
    
    print("Writing train_data\n")

    train_data_file_path = dest_path_prefix + "preprocessed_datasets/Criteo/Criteo_train_NR_DPUS_{}_label_0_{}_label_1_{}_total_{}.txt".format(NR_DPUS, total_number_label_0, total_number_label_1, total_number)
    with open(train_data_file_path, 'w+') as f:
        f.writelines(train_data)

    print("End writing train_data\n")
    exit(0)



















