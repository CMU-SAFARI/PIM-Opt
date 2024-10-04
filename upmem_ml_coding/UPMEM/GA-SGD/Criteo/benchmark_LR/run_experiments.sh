#!/bin/bash

NR_DPUS=(2048) 
NR_TASKLETS=(16)
AE=1 ## 1 corrensponds to running Artifact Evaluation; replace with 0 to manually set hyperparameters
DEST_DIR="/local-data/upmem0013/AE_PACT_2024"
SOURCE_DIR="${DEST_DIR}/preprocessed_datasets/Criteo"
alg="GA-SGD"
model="LR"


for j in "${NR_TASKLETS[@]}"; do
    for i in "${NR_DPUS[@]}"; do
        log_file="benchmark__${model}__${alg}__Criteo__NR_DPUS_${i}_NR_TASKLETS_${j}.log"
        start_time=$(date +%s)
        start_time_human=$(date)
        {
            echo "Start time: $start_time_human"
            echo "Running with NR_DPUS=$i, and NR_TASKLETS=$j"

            sleep 30

            # Compile
            NR_DPUS=$i NR_TASKLETS=$j AE=$AE SOURCE_DIR=$SOURCE_DIR DEST_DIR=$DEST_DIR make

            sleep 30

            # Run the experiment
            ./bin/host_code

            sleep 30

            # Clean up and remove bin directory
            rm -r bin

            end_time=$(date +%s)
            end_time_human=$(date)
            echo "End time: $end_time_human"
            duration=$((end_time - start_time))

            hours=$((duration / 3600))
            minutes=$(((duration % 3600) / 60))
            seconds=$((duration % 60))

            echo "runtime: $hours h $minutes min $seconds s"
        } > "$log_file" 2>&1

        echo "benchmark__${model}__${alg}__Criteo__NR_DPUS_${i}_NR_TASKLETS_${j}, runtime: $hours h $minutes min $seconds s" | tee -a "$log_file"

    done
done