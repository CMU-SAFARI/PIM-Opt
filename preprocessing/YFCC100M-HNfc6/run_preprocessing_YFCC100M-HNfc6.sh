#!/bin/bash

SOURCE_DIR=$1
PARTITION=$2
NODE=$3
SOURCE_DIR_ORIGINAL="${SOURCE_DIR}original_datasets/YFCC100M-HNfc6/"

log_file="${SOURCE_DIR}logs/AE_preprocessing_YFCC100M.log"
sub_log_file="${SOURCE_DIR}logs/SUB_LOG_preprocessing_YFCC100M.log"

total_start_time=$(date +%s)

log() {
    echo "$1" | tee -a "$log_file" "$sub_log_file"
}

time_substep() {
    local start_time=$(date +%s)
    log "$1"
    eval "$2"
    wait
    local end_time=$(date +%s)
    local runtime=$((end_time - start_time))
    local hours=$((runtime / 3600))
    local minutes=$(((runtime % 3600) / 60))
    local seconds=$((runtime % 60))
    log "$1 completed in $hours h $minutes min $seconds s"
}

wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 1
    done
}

log "Start preprocessing YFCC100M-HNfc6 dataset"

max_jobs=35
time_substep "Loading tags from original YFCC100M-HNfc6 dataset" '
for value in {0..96}; do
   wait_for_jobs
   PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 preprocess_yfcc100m.py $value $SOURCE_DIR_ORIGINAL >> "$sub_log_file" 2>&1 &
done
wait
'

time_substep "Initial preprocessing: extract samples with tags indoor or with tags outdoor and store as floating point data format" '
for value in {0..96}; do
    PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 preprocess_original_data_extract_label_0_label_1_store_float_data_format.py $value $SOURCE_DIR_ORIGINAL $SOURCE_DIR >> "$log_file" 2>&1 &
done
wait
'

time_substep "Combining all samples with tag indoor and store" '
PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 merge_files_label_0_label_1.py 0 $SOURCE_DIR >> "$sub_log_file" 2>&1 &
wait
'

time_substep "Combining all samples with tag outdoor and store" '
PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 merge_files_label_0_label_1.py 1 $SOURCE_DIR >> "$sub_log_file" 2>&1 &
wait
'

time_substep "Get min and max values per feature for normalization" '
PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 get_min_max_per_feature_for_normalization_parallelized.py $SOURCE_DIR >> "$sub_log_file" 2>&1 &
wait
'

time_substep "Randomly subsample YFCC100M-HNfc6 dataset" '
PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 randomly_subsample_yfcc100m_float_tag_indoor_label_0.py $SOURCE_DIR >> "$sub_log_file" 2>&1 &
PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 randomly_subsample_yfcc100m_float_tag_outdoor_label_1.py $SOURCE_DIR >> "$sub_log_file" 2>&1 &
wait
'

time_substep "Normalization of training data and test data" '
for value in {1..4}; do
    PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 normalize_train_test_data_float.py $value $SOURCE_DIR >> "$sub_log_file" 2>&1 &
done
wait
'

time_substep "Quantization of training data and test data" '
for value in {1..4}; do
    PYTHONUNBUFFERED=1 srun -p $PARTITION -w $NODE python3 quantization_train_test_data_uint32.py $value $SOURCE_DIR >> "$sub_log_file" 2>&1 &
done
wait
'

time_substep "Generating train and test data float" '
chmod +x generate_train_test_data_float.sh
srun -p $PARTITION -w $NODE ./generate_train_test_data_float.sh $SOURCE_DIR >> "$sub_log_file" 2>&1 
wait
'

time_substep "Generating train and test data uint32" '
chmod +x generate_train_test_data_uint32.sh
srun -p $PARTITION -w $NODE ./generate_train_test_data_uint32.sh $SOURCE_DIR >> "$sub_log_file" 2>&1 
wait
'

time_substep "Convert float to pt data format for baselines" '
srun -p $PARTITION -w $NODE python3 converting_float_to_pt_data_format_for_baselines.py $SOURCE_DIR >> "$sub_log_file" 2>&1 
wait
'

log "Done preprocessing YFCC100M-HNfc6 dataset"

total_end_time=$(date +%s)
total_runtime=$((total_end_time - total_start_time))
total_hours=$((total_runtime / 3600))
total_minutes=$(((total_runtime % 3600) / 60))
total_seconds=$((total_runtime % 60))

total_message="Total runtime = $total_hours h $total_minutes min $total_seconds s"
log "$total_message"

log "SUCCESS"
