#!/bin/bash
source_path_tmp=$1
root="${source_path_tmp}preprocessing/YFCC100M-HNfc6/preprocessed_normalized_float"
dst_path="${source_path_tmp}preprocessed_datasets/YFCC100M-HNfc6/float"


file_0="normalized_train_randomly_subsampled_number_of_samples_5000000_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_float.txt"
file_1="normalized_train_randomly_subsampled_number_of_samples_5000000_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_float.txt"
file_test_0="normalized_test_randomly_subsampled_number_of_samples_1000000_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_float.txt"
file_test_1="normalized_test_randomly_subsampled_number_of_samples_1000000_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_float.txt"


# NR_DPUs = 2048
(cat <(head -n 3407872 "${root}/${file_0}") <(head -n 3407872 "${root}/${file_1}") | shuf > "${dst_path}/YFCC100M_train_corresponding_to_NR_DPUs_2048_label_0_3407872_label_1_3407872_total_6815744_float.txt") 


# NR_DPUs = 2048
(cat <(head -n 851968 "${root}/${file_test_0}") <(head -n 851968 "${root}/${file_test_1}") | shuf > "${dst_path}/YFCC100M_test_corresponding_to_NR_DPUs_2048_label_0_851968_label_1_851968_total_1703936_float.txt") 







