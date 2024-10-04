#!/bin/bash

SOURCE_DIR=$1 
PARTITION=$2 
NODE=$3 
BASE_DIR="results/results_UPMEM"

mkdir -p "${SOURCE_DIR}logs"
log_file="${SOURCE_DIR}logs/AE_postprocessing_Criteo.log"
sub_log_file="${SOURCE_DIR}logs/SUB_LOG_postprocessing_Criteo.log"

declare -a result_dirs=(
    "/Criteo/strong_scaling/admm_LR_uint32"
    "/Criteo/strong_scaling/admm_SVM_uint32"
    "/Criteo/strong_scaling/ga_LR_uint32"
    "/Criteo/strong_scaling/ga_SVM_uint32"
    "/Criteo/strong_scaling/mbsgd_LR_uint32"
    "/Criteo/strong_scaling/mbsgd_SVM_uint32"
    "/Criteo/weak_scaling/admm_LR_uint32"
    "/Criteo/weak_scaling/admm_SVM_uint32"
    "/Criteo/weak_scaling/ga_LR_uint32"
    "/Criteo/weak_scaling/ga_SVM_uint32"
    "/Criteo/weak_scaling/mbsgd_LR_uint32"
    "/Criteo/weak_scaling/mbsgd_SVM_uint32"
)

total_start_time=$(date +%s)

log() {
    echo "$1" | tee -a "$log_file" "$sub_log_file"
}

wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 1
    done
}

log "Start postprocessing Criteo dataset (computing ROC AUC Score)"

max_jobs=3

for dir in "${result_dirs[@]}"; do
    mkdir -p "${SOURCE_DIR}postprocessed_results_UPMEM${dir}"
    for NR_DPUs in {256,512,1024,2048}; do
        if [[ "$dir" == *"strong"* && $NR_DPUs -eq 256 ]]; then
            continue
        fi
        wait_for_jobs
        PYTHONUNBUFFERED=1 srun -c 32 -p $PARTITION -w $NODE python3 postprocessing_results_UPMEM_Criteo.py $NR_DPUs "${SOURCE_DIR}${BASE_DIR}${dir}" "${SOURCE_DIR}postprocessed_results_UPMEM${dir}" "$SOURCE_DIR" >> "$sub_log_file" 2>&1 &
        
        log "Postprocessing files in ../${BASE_DIR}${dir}, NR_DPUs = $NR_DPUs"
    done
done




log "Start postprocessing Criteo dataset (computing ROC AUC Score)"

total_end_time=$(date +%s)
total_runtime=$((total_end_time - total_start_time))
total_hours=$((total_runtime / 3600))
total_minutes=$(((total_runtime % 3600) / 60))
total_seconds=$((total_runtime % 60))

total_message="Total runtime = $total_hours h $total_minutes min $total_seconds s"
log "$total_message"

log "SUCCESS"