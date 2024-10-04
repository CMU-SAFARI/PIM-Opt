#!/bin/bash

ROOT_DIR=${1}
PARTITION_CPU=${2}
NODE_CPU=${3}

### Run the training scripts ###
srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 train_criteo.py --device cpu --dataset criteo \
--path ${ROOT_DIR}/Criteo/Criteo_train_NR_DPUS_2048_label_0_388882440_label_1_13770744_total_402653184.txt \
--dataset_size 402653184 \
--model lr --optim sgd --dist_type ma1

srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 train_criteo.py --device cpu --dataset criteo \
--path ${ROOT_DIR}/Criteo/Criteo_train_NR_DPUS_2048_label_0_388882440_label_1_13770744_total_402653184.txt \
--dataset_size 402653184 \
--model svm --optim sgd --dist_type ma1

srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 train_criteo.py --device cpu --dataset criteo \
--path ${ROOT_DIR}/Criteo/Criteo_train_NR_DPUS_2048_label_0_388882440_label_1_13770744_total_402653184.txt \
--dataset_size 402653184 \
--model lr --optim admm --dist_type ma1

srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 train_criteo.py --device cpu --dataset criteo \
--path ${ROOT_DIR}/Criteo/Criteo_train_NR_DPUS_2048_label_0_388882440_label_1_13770744_total_402653184.txt \
--dataset_size 402653184 \
--model svm --optim admm --dist_type ma1

srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 train_criteo.py --device cpu --dataset criteo \
--path ${ROOT_DIR}/Criteo/Criteo_train_NR_DPUS_2048_label_0_388882440_label_1_13770744_total_402653184.txt \
--dataset_size 402653184 \
--model lr --optim sgd --dist_type ga

srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 train_criteo.py --device cpu --dataset criteo \
--path ${ROOT_DIR}/Criteo/Criteo_train_NR_DPUS_2048_label_0_388882440_label_1_13770744_total_402653184.txt \
--dataset_size 402653184 \
--model svm --optim sgd --dist_type ga

### Run the evaluation scripts ###
mkdir -p ./data/criteo/lr
mkdir -p ./data/criteo/svm
mv ./cpu_criteo_lr*.pt ./data/criteo/lr/
mv ./cpu_criteo_svm*.pt ./data/criteo/svm/

srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 eval.py --model lr --dataset criteo --path ${ROOT_DIR}/Criteo/ --epoch_data_path ./data/criteo/lr --output ./baseline_criteo_lr.csv --num_procs 128
srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 eval.py --model svm --dataset criteo --path ${ROOT_DIR}/Criteo/ --epoch_data_path ./data/criteo/svm --output ./baseline_criteo_svm.csv --num_procs 128

