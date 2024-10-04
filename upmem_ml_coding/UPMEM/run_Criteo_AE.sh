#!/bin/bash

DEST_DIR=$1
CODE_DIR="${DEST_DIR}/upmem_ml_coding/UPMEM"
SOURCE_DIR="${DEST_DIR}/preprocessed_datasets/Criteo"


# Base directory
base_dir="results_UPMEM"

mkdir -p "$DEST_DIR/$base_dir/Criteo/strong_scaling/admm_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/strong_scaling/admm_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/strong_scaling/ga_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/strong_scaling/ga_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/strong_scaling/mbsgd_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/strong_scaling/mbsgd_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/weak_scaling/admm_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/weak_scaling/admm_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/weak_scaling/ga_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/weak_scaling/ga_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/weak_scaling/mbsgd_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/weak_scaling/mbsgd_SVM_uint32"


# Base directory
base_dir="benchmark_UPMEM"

mkdir -p "$DEST_DIR/$base_dir/Criteo/admm_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/admm_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/ga_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/ga_SVM_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/mbsgd_LR_uint32"
mkdir -p "$DEST_DIR/$base_dir/Criteo/mbsgd_SVM_uint32"


log_file="$DEST_DIR/logs/run_Criteo_all_experiments.log"
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
                message="Start running experiments Criteo $a $m"
            else
                message="Start running benchmarks Criteo $a $m"
            fi
            echo "$message" | tee -a "$log_file"

            if [ "$e" -eq 0 ]; then
                cd "$CODE_DIR/$a/Criteo/$m"
            else
                cd "$CODE_DIR/$a/Criteo/benchmark_$m"
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
                message="Done running experiments Criteo $a $m, runtime is $hours h $minutes min $seconds s"
            else
                message="Done running benchmarks Criteo $a $m, runtime is $hours h $minutes min $seconds s"
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

total_message="Done experiments and benchmarks Criteo, total runtime = $total_hours h $total_minutes min $total_seconds s"
echo "$total_message" | tee -a "$log_file"