#!/bin/bash

NR_DPUS=(256 512 1024 2048)
NR_TASKLETS=(16) 
SCALING=(0 1) ## 0 corresponds to weak scaling and 1 corresponds to strong scaling
AE=1 ## 1 corrensponds to running Artifact Evaluation; replace with 0 to manually set hyperparameters
SOURCE_DIR=$1
DEST_DIR=$2
top_log_file=$3
alg=$4
model=$5

# Iterate through each value
for j in "${NR_TASKLETS[@]}"; do
    for i in "${NR_DPUS[@]}"; do
        for s in "${SCALING[@]}"; do
            # Skip if SCALING is 1 and NR_DPUS is 256 
            if [ "$s" -eq 1 ] && [ "$i" -eq 256 ] && [ "$AE" -eq 1 ]; then
                continue
            fi
            log_file="$DEST_DIR/logs/${model}__${alg}__YFCC100M-HNfc6__NR_DPUS_${i}_NR_TASKLETS_${j}_SCALING_${s}.log"
            start_time=$(date +%s)
            start_time_human=$(date)
            {
                echo "Start time: $start_time_human"
                echo "Running with NR_DPUS=$i, NR_TASKLETS=$j, and SCALING=$s"

                sleep 30

                # Compile
                NR_DPUS=$i NR_TASKLETS=$j SCALING=$s AE=$AE SOURCE_DIR=$SOURCE_DIR DEST_DIR=$DEST_DIR make

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

            echo "${model}__${alg}__YFCC100M-HNfc6__NR_DPUS_${i}_NR_TASKLETS_${j}_SCALING_${s}, runtime: $hours h $minutes min $seconds s" | tee -a "$top_log_file"

        done
    done
done
