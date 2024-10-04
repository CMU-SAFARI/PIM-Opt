#!/bin/bash
source_path_tmp=$1
root="${source_path_tmp}preprocessing/YFCC100M-HNfc6/preprocessed_quantization_uint32"
dst_path="${source_path_tmp}preprocessed_datasets/YFCC100M-HNfc6/uint32"

file_0="train_randomly_subsampled_number_of_samples_5000000_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_uint32.txt"
file_1="train_randomly_subsampled_number_of_samples_5000000_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_uint32.txt"
file_test_0="test_randomly_subsampled_number_of_samples_1000000_YFCC100M_hybridCNN_gmean_fc6_tag_indoor_label_0_features_of_type_uint32.txt"
file_test_1="test_randomly_subsampled_number_of_samples_1000000_YFCC100M_hybridCNN_gmean_fc6_tag_outdoor_label_1_features_of_type_uint32.txt"



# NR_DPUs = 2048
(cat <(head -n 3407872 "${root}/${file_0}") <(head -n 3407872 "${root}/${file_1}") | shuf > "${dst_path}/YFCC100M_train_corresponding_to_NR_DPUs_2048_label_0_3407872_label_1_3407872_total_6815744_uint32.txt") &

# NR_DPUs = 2048
(cat <(head -n 851968 "${root}/${file_test_0}") <(head -n 851968 "${root}/${file_test_1}") | shuf > "${dst_path}/YFCC100M_test_corresponding_to_NR_DPUs_2048_label_0_851968_label_1_851968_total_1703936_uint32.txt") &

wait

# NR_DPUs = 1024
(cat <(head -n 1703936 "${root}/${file_0}") <(head -n 1703936 "${root}/${file_1}") | shuf > "${dst_path}/YFCC100M_train_corresponding_to_NR_DPUs_1024_label_0_1703936_label_1_1703936_total_3407872_uint32.txt") &

# NR_DPUs = 1024
(cat <(head -n 425984 "${root}/${file_test_0}") <(head -n 425984 "${root}/${file_test_1}") | shuf > "${dst_path}/YFCC100M_test_corresponding_to_NR_DPUs_1024_label_0_425984_label_1_425984_total_851968_uint32.txt") &

wait

# NR_DPUs = 512
(cat <(head -n 851968 "${root}/${file_0}") <(head -n 851968 "${root}/${file_1}") | shuf > "${dst_path}/YFCC100M_train_corresponding_to_NR_DPUs_512_label_0_851968_label_1_851968_total_1703936_uint32.txt") &

# NR_DPUs = 512
(cat <(head -n 212992 "${root}/${file_test_0}") <(head -n 212992 "${root}/${file_test_1}") | shuf > "${dst_path}/YFCC100M_test_corresponding_to_NR_DPUs_512_label_0_212992_label_1_212992_total_425984_uint32.txt") &

wait

# NR_DPUs = 256
(cat <(head -n 425984 "${root}/${file_0}") <(head -n 425984 "${root}/${file_1}") | shuf > "${dst_path}/YFCC100M_train_corresponding_to_NR_DPUs_256_label_0_425984_label_1_425984_total_851968_uint32.txt") &

# NR_DPUs = 256
(cat <(head -n 106496 "${root}/${file_test_0}") <(head -n 106496 "${root}/${file_test_1}") | shuf > "${dst_path}/YFCC100M_test_corresponding_to_NR_DPUs_256_label_0_106496_label_1_106496_total_212992_uint32.txt") &

wait