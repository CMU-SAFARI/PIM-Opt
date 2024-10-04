#!/bin/bash

SOURCE_DIR=$1
PARTITION=$2
NODE=$3
SOURCE_DIR_ORIGINAL="${SOURCE_DIR}original_datasets/Criteo/"


log_file="${SOURCE_DIR}logs/AE_preprocessing_Criteo.log"
sub_log_file="${SOURCE_DIR}logs/SUB_LOG__preprocessing_Criteo.log"


total_start_time=$(date +%s)

log() {
    echo "$1" | tee -a "$log_file" "$sub_log_file"
}



log "Start preprocessing Criteo dataset"
for value in {0..23}; do
    
    PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 gather_statistics_and_preprocess_label_0_label_1.py $value $SOURCE_DIR_ORIGINAL $SOURCE_DIR >> "$sub_log_file" 2>&1 &

    log "Launched job $value"

done

wait

log "Start subsampling Criteo dataset"

for NR_DPUs in {256,512,1024,2048}; do
    PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 shuffle_and_subsample_per_day_and_combine_all.py $NR_DPUs $SOURCE_DIR >> "$sub_log_file" 2>&1 &
    
    log "Processing dataset corresponding to NR_DPUs = $NR_DPUs "
    wait
done


shuf "${SOURCE_DIR}preprocessed_datasets/Criteo/test_data_criteo_tb_day_23.txt" > "${SOURCE_DIR}preprocessed_datasets/Criteo/test_data_criteo_tb_day_23_shuffled.txt"

mv "${SOURCE_DIR}preprocessed_datasets/Criteo/test_data_criteo_tb_day_23_shuffled.txt" "${SOURCE_DIR}preprocessed_datasets/Criteo/test_data_criteo_tb_day_23.txt"


log "Done preprocessing Criteo dataset"

total_end_time=$(date +%s)
total_runtime=$((total_end_time - total_start_time))
total_hours=$((total_runtime / 3600))
total_minutes=$(((total_runtime % 3600) / 60))
total_seconds=$((total_runtime % 60))

total_message="Total runtime = $total_hours h $total_minutes min $total_seconds s"
log "$total_message"

log "SUCCESS"
