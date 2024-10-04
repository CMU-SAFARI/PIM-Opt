#!/bin/bash

SOURCE_DIR=$1 
PARTITION=$2 
NODE=$3 

mkdir -p "${SOURCE_DIR}logs"
mkdir -p "${SOURCE_DIR}preprocessing/YFCC100M-HNfc6/initial_preprocessing"
mkdir -p "${SOURCE_DIR}preprocessing/YFCC100M-HNfc6/preprocessed_label_0_label_1_float"
mkdir -p "${SOURCE_DIR}preprocessing/YFCC100M-HNfc6/preprocessed_normalized_float"
mkdir -p "${SOURCE_DIR}preprocessing/YFCC100M-HNfc6/preprocessed_quantization_uint32"
mkdir -p "${SOURCE_DIR}preprocessing/YFCC100M-HNfc6/preprocessed_randomly_subsampled"
cd "${SOURCE_DIR}preprocessing/YFCC100M-HNfc6"
chmod +x "run_preprocessing_YFCC100M-HNfc6.sh"

./run_preprocessing_YFCC100M-HNfc6.sh $SOURCE_DIR $PARTITION $NODE &

wait


mkdir -p "${SOURCE_DIR}preprocessing/Criteo/preprocessed_label_0_label_1"
mkdir -p "${SOURCE_DIR}preprocessing/Criteo/statistics"
cd "${SOURCE_DIR}preprocessing/YFCC100M-HNfc6"
chmod +x "run_preprocessing_YFCC100M-HNfc6.sh"

./run_preprocessing_Criteo.sh $SOURCE_DIR $PARTITION $NODE &

wait

echo "Preprocessing completed."