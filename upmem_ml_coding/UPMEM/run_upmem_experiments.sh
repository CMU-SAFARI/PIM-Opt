#!/bin/bash


ROOT_DIR=$1

chmod +x run_YFCC100M-HNfc6_AE.sh
./run_YFCC100M-HNfc6_AE.sh $ROOT_DIR &

wait

chmod +x run_Criteo_AE.sh
./run_Criteo_AE.sh $ROOT_DIR &
wait

echo "Done with all experiments on the UPMEM PIM System"