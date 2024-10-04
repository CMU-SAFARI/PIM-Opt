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
    

    file_0 = "all_files_combined_number_of_samples_21608224_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_float.txt"
    

    k = 6000000
    l = 5000000
    
    os.chdir(source_dir)

    random.seed(seed_0)
    sample_indices_0 = random.sample(range(21608224), k)
    sample_indices_0.sort()


    

    output_lines_0 = []

    print("start file_0\n")
    with open(file_0, "r") as infile:
        for current_index, line in enumerate(infile):
            if len(sample_indices_0) == 0:
                break
            if current_index == sample_indices_0[0]:
                output_lines_0.append(line)
                del sample_indices_0[0]


    print("end file_0\n")
   
    random.seed(seed_2)
    random.shuffle(output_lines_0)
    print("shuffled 0\n")

    
    os.chdir(dest_dir)

    output_lines_test = output_lines_0[l:]
    with open('test_randomly_subsampled_number_of_samples_1000000_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_float.txt', 'w+') as f:
        f.writelines(output_lines_test)

    print("end writing test output\n")

    output_lines_0 = output_lines_0[:l]
    
    with open('train_randomly_subsampled_number_of_samples_5000000_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_float.txt', 'w+') as f:
        f.writelines(output_lines_0)
    print("end writing train output 0\n")

    






