#!/bin/bash

DEST_DIR=$1
CODE_DIR="${DEST_DIR}/upmem_ml_coding/UPMEM"
SOURCE_DIR="${DEST_DIR}/preprocessed_datasets/YFCC100M-HNfc6"


# Base directory
base_dir="results_UPMEM"

mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/strong_scaling/admm_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/strong_scaling/admm_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/strong_scaling/ga_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/strong_scaling/ga_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/strong_scaling/mbsgd_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/strong_scaling/mbsgd_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/weak_scaling/admm_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/weak_scaling/admm_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/weak_scaling/ga_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/weak_scaling/ga_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/weak_scaling/mbsgd_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/weak_scaling/mbsgd_SVM_uint32"


# Base directory
base_dir="benchmark_UPMEM"

mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/admm_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/admm_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/ga_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/ga_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/mbsgd_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/YFCC100M-HNfc6/mbsgd_SVM_uint32"

# Log directory
mkdir -p "$DEST_DIR/logs"


log_file="$DEST_DIR/logs/AE_YFCC100M-HNfc6_main_log_file.log"
alg=("MA-SGD" "GA-SGD" "ADMM")
model=("LR" "SVM")
experiment_type=(1 0) ## 0 corresponds to running experiments, 1 corresponds to running benchmarks

echo "Directory structure created successfully." | tee -a "$log_file"

# Start total runtime measurement
total_start_time=$(date +%s)

for a in "${alg[@]}"; do
    for m in "${model[@]}"; do
        for e in "${experiment_type[@]}"; do
            start_time=$(date +%s)
            start_time_human=$(date)
            
            if [ "$e" -eq 0 ]; then
                message="Start running experiments YFCC100M-HNfc6 $a $m"
            else
                message="Start running benchmarks YFCC100M-HNfc6 $a $m"
            fi
            echo "$message" | tee -a "$log_file"

            if [ "$e" -eq 0 ]; then
                cd "$CODE_DIR/$a/YFCC100M-HNfc6/$m"
            else
                cd "$CODE_DIR/$a/YFCC100M-HNfc6/benchmark_$m"
            fi
            chmod +x run_experiments.sh
            ./run_experiments.sh "$SOURCE_DIR" "$DEST_DIR" "$log_file" "$a" "$m"
            
            end_time=$(date +%s)
            end_time_human=$(date)
            runtime=$((end_time - start_time))
            hours=$((runtime / 3600))
            minutes=$(((runtime % 3600) / 60))
            seconds=$((runtime % 60))

            if [ "$e" -eq 0 ]; then
                message="Done running experiments YFCC100M-HNfc6 $a $m, runtime is $hours h $minutes min $seconds s"
            else
                message="Done running benchmarks YFCC100M-HNfc6 $a $m, runtime is $hours h $minutes min $seconds s"
            fi
            echo "$message" | tee -a "$log_file"
        done
    done
done

# End total runtime measurement
total_end_time=$(date +%s)
total_runtime=$((total_end_time - total_start_time))
total_hours=$((total_runtime / 3600))
total_minutes=$(((total_runtime % 3600) / 60))
total_seconds=$((total_runtime % 60))

total_message="Done experiments and benchmarks YFCC100M-HNfc6, total runtime = $total_hours h $total_minutes min $total_seconds s"
echo "$total_message" | tee -a "$log_file"