import os
import sys
import numpy as np
from itertools import islice


def initialize(kind):
    init_list = []
    add_bound = 0
    if kind == "min":
        add_bound = 1000000
    for i in range(0, 39):
        init_list.append(int(0+add_bound))
    return init_list

def max_min_bounds(min_list, max_list, cat_list):
    cat_list_int = [int(x) for x in cat_list[:]]
    for i in range(0,39):
        if cat_list_int[i] < min_list[i]:
            min_list[i] = cat_list_int[i]
        if cat_list_int[i] > max_list[i]:
            max_list[i] = cat_list_int[i]
    return min_list, max_list


def preprocess(filename_path, filename_path_0, filename_path_1, statistics_path, start_index, number_of_samples_job_id, job_id):
    num_lines_processed = 0

    count_0 = 0
    count_1 = 0

    feat_scaling_different_0 = 0
    feat_scaling_different_1 = 0

    min_features_bounds_0 = initialize("min")
    max_features_bounds_0 = initialize("max")
    min_features_bounds_1 = initialize("min")
    max_features_bounds_1 = initialize("max")
    
    file = open(filename_path, "r")
    
    file_0 = open(filename_path_0, "w+")
    file_1 = open(filename_path_1, "w+")

    for line in islice(file, start_index):
        pass
    
    for k in range(0, number_of_samples_job_id):
        if num_lines_processed >= 100000000:
            if num_lines_processed % 100000000 == 0:
                print("job_id {}: # Processed lines = {}\n".format(job_id, num_lines_processed))
                print("job_id {}: # label_0 = {}\n".format(job_id, count_0))
                print("job_id {}: # label_1 = {}\n".format(job_id, count_1))
                print("job_id {}: # different feature scaling for samples label 0 in LIBSVM format = {}\n".format(job_id, feat_scaling_different_0))
                print("job_id {}: # different feature scaling for samples label 1 in LIBSVM format = {}\n\n".format(job_id, feat_scaling_different_1))
        elif num_lines_processed >= 10000000:
            if num_lines_processed % 10000000 == 0:
                print("job_id {}: # Processed lines = {}\n".format(job_id, num_lines_processed))
                print("job_id {}: # label_0 = {}\n".format(job_id, count_0))
                print("job_id {}: # label_1 = {}\n".format(job_id, count_1))
                print("job_id {}: # different feature scaling for samples label 0 in LIBSVM format = {}\n".format(job_id, feat_scaling_different_0))
                print("job_id {}: # different feature scaling for samples label 1 in LIBSVM format = {}\n\n".format(job_id, feat_scaling_different_1))
        elif num_lines_processed >= 1000000:
            if num_lines_processed % 1000000 == 0:
                print("job_id {}: # Processed lines = {}\n".format(job_id, num_lines_processed))
                print("job_id {}: # label_0 = {}\n".format(job_id, count_0))
                print("job_id {}: # label_1 = {}\n".format(job_id, count_1))
                print("job_id {}: # different feature scaling for samples label 0 in LIBSVM format = {}\n".format(job_id, feat_scaling_different_0))
                print("job_id {}: # different feature scaling for samples label 1 in LIBSVM format = {}\n\n".format(job_id, feat_scaling_different_1))
        elif num_lines_processed >= 100000:
            if num_lines_processed % 100000 == 0:
                print("job_id {}: # Processed lines = {}\n".format(job_id, num_lines_processed))
                print("job_id {}: # label_0 = {}\n".format(job_id, count_0))
                print("job_id {}: # label_1 = {}\n".format(job_id, count_1))
                print("job_id {}: # different feature scaling for samples label 0 in LIBSVM format = {}\n".format(job_id, feat_scaling_different_0))
                print("job_id {}: # different feature scaling for samples label 1 in LIBSVM format = {}\n\n".format(job_id, feat_scaling_different_1))

        line = file.readline()
        if not line:
            break
        splits = line.split()
        if len(splits) == 40:
            label = splits[0]
            cat_features = []
            for i in range(1,40):
                feature_splits = splits[i].split(":")
                if feature_splits[1] != "0.16013":
                    if label == "0":
                        feat_scaling_different_0 += 1
                    else:
                        feat_scaling_different_1 += 1
                else:
                    cat_features.append(feature_splits[0])
            if len(cat_features) == 39:
                out_str = label + " " + " ".join(cat_features) + "\n"
                if label == "0":
                    count_0 += 1
                    file_0.write(out_str)
                    min_features_bounds_0, max_features_bounds_0 = max_min_bounds(min_features_bounds_0, max_features_bounds_0, cat_features)
                else:
                    count_1 += 1
                    file_1.write(out_str)
                    min_features_bounds_1, max_features_bounds_1 = max_min_bounds(min_features_bounds_1, max_features_bounds_1, cat_features)
        num_lines_processed += 1

    statistics_file = open(statistics_path, "w+")
    statistics_file.write("# Processed lines = {}\n".format(num_lines_processed))
    statistics_file.write("# label_0 = {}\n".format(count_0))
    statistics_file.write("# label_1 = {}\n".format(count_1))
    statistics_file.write("# different feature scaling for samples label 0 in LIBSVM format = {}\n".format(feat_scaling_different_0))
    statistics_file.write("# different feature scaling for samples label 1 in LIBSVM format = {}\n\n".format(feat_scaling_different_1))

    min_overall_features_bounds = initialize("min")
    max_overall_features_bounds = initialize("max")
    for i in range(0, 39):
        if int(min_features_bounds_0[i]) < int(min_features_bounds_1[i]):
            min_overall_features_bounds[i] = int(min_features_bounds_0[i])
        else:
            min_overall_features_bounds[i] = int(min_features_bounds_1[i])
        if int(max_features_bounds_0[i]) > int(max_features_bounds_1[i]):
            max_overall_features_bounds[i] = int(max_features_bounds_0[i])
        else:
            max_overall_features_bounds[i] = int(max_features_bounds_1[i])
        
    statistics_file.write("Overall: Categorical features bounds\n")
    for i in range(0, 39):
        statistics_file.write("{},{}\n".format(min_overall_features_bounds[i], max_overall_features_bounds[i]))
    statistics_file.write("\n")

    statistics_file.write("Label 0: Categorical features bounds\n")
    for i in range(0, 39):
        statistics_file.write("{},{}\n".format(min_features_bounds_0[i], max_features_bounds_0[i]))
    statistics_file.write("\n")
    statistics_file.write("Label 1: Categorical features bounds\n")
    for i in range(0, 39):
        statistics_file.write("{},{}\n".format(min_features_bounds_1[i], max_features_bounds_1[i]))

    statistics_file.close()
    file.close()
    file.close()
    file_0.close()
    file_1.close()

                
                 

if __name__ == "__main__":
    job_id = int(sys.argv[1])
    source_path_prefix = str(sys.argv[2])
    dest_path_prefix = str(sys.argv[3])
    number_of_samples_per_day = [195841983,199563535,196792019,181115208,152115810,172548507,204846845,200801003,193772492,198424372,185778055,153588700,169003364,194216520,194081279,187154596,177984934,163382602,142061091,156534237,193627464,192215183,189747893]

    # sanity check
    check_tmp = 0
    for i in range(0,23):
        check_tmp += number_of_samples_per_day[i]
    check_tmp += 178274637 # number of test samples
    if (check_tmp != 4373472329):
        print("Sanity check NOT passed!!!\n")
        exit(0)
    else:
        print("OK!\n")

    start_index_list = [0]
    for i in range(1,23):
        start_index_list.append(start_index_list[len(start_index_list)-1]+number_of_samples_per_day[i-1])

    start_index = 0 #start_index_list[job_id]
    number_of_samples_job_id = 0
    if job_id < 23:
        start_index = start_index_list[job_id]
        number_of_samples_job_id = number_of_samples_per_day[job_id]

    filename_path = ""
    if job_id == 23:
        filename_path = source_path_prefix + "criteo_tb.svm/criteo_tb.t"
        start_index = 0
        number_of_samples_job_id = 178274637
    else:
        filename_path = source_path_prefix + "criteo_tb.svm/criteo_tb"

    filename_path_0 = dest_path_prefix + "preprocessing/Criteo/preprocessed_label_0_label_1/criteo_tb_label_0_job_id_{}.txt".format(job_id)

    filename_path_1 = dest_path_prefix + "preprocessing/Criteo/preprocessed_label_0_label_1/criteo_tb_label_1_job_id_{}.txt".format(job_id)

    statistics_path = dest_path_prefix + "preprocessing/Criteo/statistics/statistics_parallel_job_id_{}.txt".format(job_id)

    

    preprocess(filename_path, filename_path_0, filename_path_1, statistics_path, start_index, number_of_samples_job_id, job_id)

    ## Create test data set
    if job_id == 23:
        test_file_path = dest_path_prefix + "preprocessed_datasets/Criteo/test_data_criteo_tb_day_23.txt"
        dst_file_handle = open(test_file_path, 'w+')
        for src_name in [filename_path_0, filename_path_1]:
            src_file_handle = open(src_name, 'r')
            while True:
                line = src_file_handle.readline()
                if not line:
                    break
                dst_file_handle.write(str(line))
            src_file_handle.close()
        dst_file_handle.close()

    print("Done job_id {}".format(job_id))




