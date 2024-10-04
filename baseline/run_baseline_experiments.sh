#!/bin/bash

ROOT_DIR=${1}
PARTITION_CPU=${2}
NODE_CPU=${3}
NODE_GPU=${4}

chmod +x run_criteo.sh
chmod +x run_yfcc100m.sh
./run_criteo.sh ${ROOT_DIR} ${PARTITION_CPU} ${NODE_CPU}
./run_yfcc100m.sh ${ROOT_DIR} ${PARTITION_CPU} ${NODE_CPU} ${NODE_GPU}
