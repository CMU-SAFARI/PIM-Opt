import os
import sys
import numpy as np

## Code adapted from http://www.deepfeatures.org/download.html
## Code adapted from https://github.com/DS3Lab/LambdaML


def merge_files(src_files, dst_file):
    dst_file = open(dst_file, 'w+')
    count = 0
    for src_name in src_files:
        src_file = open(src_name, 'r')
        for line in src_file:
            dst_file.write(line)
        src_file.close()
        print("Processed file number = {}".format(count))
        count += 1
    dst_file.close()


if __name__ == "__main__":
    job_id = int(sys.argv[1])
    source_path_tmp = str(sys.argv[2])

    source_path_prefix = source_path_tmp + "preprocessing/YFCC100M-HNfc6/initial_preprocessing/"
    dest_path_prefix = source_path_tmp + "preprocessing/YFCC100M-HNfc6/preprocessed_label_0_label_1_float/"
    tag_0 = 'indoor'
    tag_1 = 'outdoor'



    out_file_path_tag_0 = dest_path_prefix + "all_files_combined_number_of_samples_21608224_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_float.txt"

    out_file_path_tag_1 = dest_path_prefix + "all_files_combined_number_of_samples_30728913_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_float.txt"


    if job_id == 0:

        src_files_list = []
        for i in range(0,97):
            print("Processing file {}".format(i), flush=True)
            src_file_path = source_path_prefix + "YFCC100M_hybridCNN_gmean_fc6_{}_tag_indoor".format(i)
            src_files_list.append(src_file_path)
        merge_files(src_files_list, out_file_path_tag_0)
    if job_id == 1:
        src_files_list = []
        for i in range(0,97):
            print("Processing file {}".format(i), flush=True)
            src_file_path = source_path_prefix + "YFCC100M_hybridCNN_gmean_fc6_{}_tag_outdoor".format(i)
            src_files_list.append(src_file_path)
        merge_files(src_files_list, out_file_path_tag_1)




