import os, sys
import numpy as np
import random 



if __name__ == "__main__":
    source_path_tmp = str(sys.argv[1])
    current_dir = os.getcwd()

    source_dir = source_path_tmp + "preprocessing/YFCC100M-HNfc6/preprocessed_label_0_label_1_float"
    dest_dir = source_path_tmp + "preprocessing/YFCC100M-HNfc6/preprocessed_randomly_subsampled"


    seed_0 = 1234
    seed_1 = 2023
    seed_2 = 6081
    seed_3 = 4510
    

    file_1 = "all_files_combined_number_of_samples_30728913_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_float.txt"

    k = 6000000
    l = 5000000
    
    os.chdir(source_dir)

    random.seed(seed_1)
    sample_indices_1 = random.sample(range(30728913), k)
    sample_indices_1.sort()


    output_lines_1 = []

    print("start file_1\n")
    with open(file_1, "r") as infile:
        for current_index, line in enumerate(infile):
            if len(sample_indices_1) == 0:
                break
            if current_index == sample_indices_1[0]:
                output_lines_1.append(line)
                del sample_indices_1[0]

    print("end file_1\n")
   

    random.seed(seed_3)
    random.shuffle(output_lines_1)
    print("shuffled 1\n")

    os.chdir(dest_dir)

    output_lines_test = output_lines_1[l:]
    with open('test_randomly_subsampled_number_of_samples_1000000_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_float.txt', 'w+') as f:
        f.writelines(output_lines_test)

    print("end writing test output\n")

    output_lines_1 = output_lines_1[:l]
    

    with open('train_randomly_subsampled_number_of_samples_5000000_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_float.txt', 'w+') as f:
        f.writelines(output_lines_1)
    print("end writing train output 1\n")







