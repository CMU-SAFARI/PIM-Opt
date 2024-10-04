#!/bin/bash

ROOT_DIR=${1}
PARTITION_CPU=${2}
NODE_CPU=${3}
NODE_GPU=${4}

### Run the training scripts ###
srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 train_yfcc_cpu.py --dataset yfcc --path ${ROOT_DIR}/YFCC100M-HNfc6/ --device cpu
srun  -p gpu_part --gres gpu:A100-80GB:1 python3 train_yfcc_gpu.py --dataset yfcc --path ${ROOT_DIR}/YFCC100M-HNfc6/ --device gpu

### Run the evaluation scripts ###
mkdir -p ./data/yfcc/lr
mkdir -p ./data/yfcc/svm
mv ./cpu_yfcc_lr*.pt ./data/yfcc/lr/
mv ./cpu_yfcc_svm*.pt ./data/yfcc/svm/
mv ./gpu_yfcc_lr*.pt ./data/yfcc/lr/
mv ./gpu_yfcc_svm*.pt ./data/yfcc/svm/

srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 eval.py --model lr --dataset yfcc --path ${ROOT_DIR}/YFCC100M-HNfc6/ --epoch_data_path ./data/yfcc/lr --output ./baseline_yfcc_lr.csv --num_procs 128
srun --exclusive -p ${PARTITION_CPU} -w ${NODE_CPU} python3 eval.py --model svm --dataset yfcc --path ${ROOT_DIR}/YFCC100M-HNfc6/ --epoch_data_path ./data/yfcc/svm --output ./baseline_yfcc_svm.csv --num_procs 128
