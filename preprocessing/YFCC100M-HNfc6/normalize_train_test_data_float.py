import os
import sys
import numpy as np


def normalize(in_name, out_file, label, min_values, diff_tmp_values):
    in_file = open(in_name, "r")

    while (True):
        line = in_file.readline()
        if not line:
            break
        
        splits = line.split()
        if len(splits) == 4097:
            splits_float = [float(x) for x in splits[1:4097]]
            splits_normalized_float = []
            for i in range(0,4096):
                splits_normalized_float.append(float((splits_float[i]-min_values[i])/diff_tmp_values[i]))
            out_str_float = str(label) + " " + " ".join(str(x) for x in splits_normalized_float[0:4096])
            out_file.write(out_str_float)
            out_file.write('\n')
            
                     
    
    in_file.close()
    






if __name__ == "__main__":
    file_number = int(sys.argv[1])
    source_path_tmp = str(sys.argv[2])
 


    label = 0
    source_file_path = ""
    src_file_path_prefix = source_path_tmp + "preprocessing/YFCC100M-HNfc6/preprocessed_randomly_subsampled/"
    out_file_path_prefix = source_path_tmp + "preprocessing/YFCC100M-HNfc6/preprocessed_normalized_float/"

    if file_number == 1:
        src_file_path = src_file_path_prefix + "test_randomly_subsampled_number_of_samples_1000000_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_float.txt"
    if file_number == 2:
        src_file_path = src_file_path_prefix + "test_randomly_subsampled_number_of_samples_1000000_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_float.txt"
        label = 1
    if file_number == 3:
        src_file_path = src_file_path_prefix + "train_randomly_subsampled_number_of_samples_5000000_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_float.txt"
    if file_number == 4:
        src_file_path = src_file_path_prefix + "train_randomly_subsampled_number_of_samples_5000000_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_float.txt"
        label = 1

    out_file_path = ""
    if file_number == 1:
        out_file_path = out_file_path_prefix + "normalized_test_randomly_subsampled_number_of_samples_1000000_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_float.txt"
    if file_number == 2:
        out_file_path = out_file_path_prefix + "normalized_test_randomly_subsampled_number_of_samples_1000000_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_float.txt"
    if file_number == 3:
        out_file_path = out_file_path_prefix + "normalized_train_randomly_subsampled_number_of_samples_5000000_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_float.txt"
    if file_number == 4:
        out_file_path = out_file_path_prefix + "normalized_train_randomly_subsampled_number_of_samples_5000000_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_float.txt"

    
    out_file = open(out_file_path, "w+")
    
    # min resp. max of all samples in yfcc100m dataset with label indoor and label outdoor but not both labels for a sample (in such case we simply ignore the sample)
    min_values = []
    max_values = []
    diff_tmp_values = []
    for i in range(0,4096):
        min_values.append(float(0))
        max_values.append(float(0))
        diff_tmp_values.append(float(0))
    
    filename = source_path_tmp + 'preprocessing/YFCC100M-HNfc6/min_max_per_feature_for_normalization_parallelized.txt'
    min_already_parsed = 0
    with open(filename, 'r') as file:
        while (True):
            line = file.readline()
            if not line:
                break
            if ',' in line:
                if min_already_parsed == 0:
                    splits = line.split(',')
                    splits_float = [float(x) for x in splits]
                    for i in range(0,4096):
                        min_values[i] = splits_float[i]
                    min_already_parsed = 1
                else:
                    splits = line.split(',')
                    splits_float = [float(x) for x in splits]
                    for i in range(0,4096):
                        max_values[i] = splits_float[i]

    for i in range(0,4096):
        diff_tmp_values[i] = max_values[i] - min_values[i]
        if diff_tmp_values[i] == float(0):
            print('ERROR divide by zero\n', flush = True)              


    
    


    normalize(src_file_path, out_file, label, min_values, diff_tmp_values)
    print("Done with file {}".format(file_number), flush = True)


    out_file.close()
