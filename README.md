# PIM-Opt: Demystifying Distributed Optimization Algorithms on a Real-World Processing-In-Memory System
This repository provides all the necessary files and instructions to reproduce the results of our [PACT'24 paper](https://arxiv.org/pdf/2404.07164v2).

> Steve Rhyner, Haocong Luo, Juan Gómez-Luna, Mohammad Sadrosadati, Jiawei Jiang, Ataberk Olgun, Harshita Gupta, Ce Zhang, and Onur Mutlu, "PIM-Opt: Demystifying Distributed Optimization Algorithms on a Real-World Processing-In-Memory System", PACT'24.

Please use the following citation to cite PIM-Opt if the repository is useful for you.
```
@inproceedings{rhyner2024pimopt,
      title={{PIM-Opt: Demystifying Distributed Optimization Algorithms on a Real-World Processing-In-Memory System}}, 
      author={Steve Rhyner, Haocong Luo, Juan Gómez-Luna, Mohammad Sadrosadati, Jiawei Jiang, Ataberk Olgun, Harshita Gupta, Ce Zhang, and Onur Mutlu},
      year={2024},
      booktitle={PACT}
}
```
Our artifact contains the source code and scripts needed to reproduce our results, including all figures in the paper. We
provide:
- source code to preprocess the YFCC100M-HNfc6 and Criteo 1TB Click Logs datasets preprocessed by LIBSVM, 
- the source code to perform experiments on the UPMEM PIM System,
- the source code of the CPU and GPU baseline implementations,
- the source code to postprocess and evaluate results, and
- Python scripts and a Jupyter Notebook to analyze and plot the results.

Please check the artifact appendix of PIM-Opt (https://arxiv.org/pdf/2404.07164v2) for a detailed description on how to reproduce the results in the paper.

## Description
### Hardware dependencies
- UPMEM PIM System: 2x Intel Xeon Silver 4215 8-core processor @ 2.50GHz, 20×8 GB UPMEM PIM modules
- CPU Baseline System: 2x AMD EPYC 7742 64-core processor @ 2.25GHz 
- GPU Baseline System: 2x Intel Xeon Gold 5118 12-core processor @ 2.30GHz, 1× NVIDIA A100 (PCIe, 80 GB) 

### Software dependencies
- `gcc (Debian 8.3.0-6) 8.3.0, GNU Make 4.2.1`
- `UPMEM SDK`, version 2023.2.0
- `tar (GNU tar) 1.34`
- `Zip 3.0`
- `Python 3.10.6`
- `pip` packages `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `coloredlogs`
- `slurm-wlm 21.08.5`
- `tmux 2.8+`
- `CUDA 11.7`

### Datasets
In this paper, we use two large-scale datasets:
- YFCC100M-HNfc6 can be requested at http://www.deepfeatures.org/index.html. For preprocessing, one needs to download the file yfcc100m_autotags.bz2 from the original YFCC100M dataset, which can be requested at https://www.multimediacommons.org.
- Criteo 1TB Click Logs preprocessed by LIBSVM can be accessed by running:
```
  $ wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/criteo_tb.svm.tar.xz
  $ tar -xJvf criteo_tb.svm.tar.xz
```

## Installation
To reproduce our results, no extra installation steps are required besides installing the dependencies as described above. We recommend using a terminal multiplexer (e.g., tmux) to ensure that experiments are completed without interruption.

## Experiment workflow
We describe the steps and commands to reproduce our results, including all figures in the paper, in this section. Note that we assume the use of slurm workload manager on a cluster. Readers with other workload managers should modify the scripts to fit their own environment.

### Preprocessing Datasets
The preprocessing of the datasets YFCC100M-HNfc6 and Criteo is initialized by running the commands:
```
  $ cd preprocessing
  $ DATA_ROOT=<path-to-data>
  $ PARTITION=<name-of-slurm-partition>
  $ NODE=<name-of-slurm-node>
  $ ./run_preprocessing.sh ${DATA_ROOT} ${PARTITION} ${NODE} &  
```

### UPMEM PIM System Experiments
To perform the experiments on the UPMEM PIM system, readers can run the command:
```
  $ cd upmem_ml_coding/UPMEM
  $ DATA_ROOT=<path-to-data>
  $ ./run_upmem_experiments.sh ${DATA_ROOT} &
```

### CPU and GPU Baseline Experiments
The baseline experiments are launched by running the commands:
```
  $ cd baseline
  $ DATA_ROOT=<path-to-data>
  $ PARTITION=<name-of-slurm-partition>
  $ NODE_CPU=<name-of-slurm-cpu_node>
  $ NODE_GPU=<name-of-slurm-gpu_node>
  $ ./run_baseline_experiments.sh ${DATA_ROOT} ${PARTITION} ${NODE_CPU} ${NODE_GPU} &
```

### Postprocessing Results
Before continuing, the `UPMEM PIM System Experiments` and the `CPU and GPU Baseline Experiments` must be completed. To continue with the postprocessing of the UPMEM PIM system results, i.e., computing metrics such as AUC Score, place the UPMEM PIM system results into the directory `/results`. Next, please run the commands:
```
  $ cd postprocessing
  $ DATA_ROOT=<path-to-data>
  $ PARTITION=<name-of-slurm-partition>
  $ NODE=<name-of-slurm-node>
  $ ./run_postprocessing_Criteo.sh ${DATA_ROOT} ${PARTITION} ${NODE} &
```

### Reproducing Figures
Please navigate to the directory `/paper_plots`, open the Jupyter Notebook `paper_plots.ipynb`, and select `Run All` or if you prefer, you can click through the Jupyter Notebook cell by cell. The generated figures can be viewed at `/paper_plots/output` in `pdf` and `png` format.

## Evaluation and expected results
Running the experiments described in `Experiment workflow` is sufficient to reproduce all of our results (`Fig. 2`, `Fig. 4`, `Fig. 5`, `Fig. 6`, `Fig. 7`, `Fig. 8`, `Fig. 9`, `Fig. 10`, `Fig. 11`, `Fig. 12`, and `Fig. 13`).


## Disclaimer
This repository contains reused and repurposed code from:
- PrIM (Processing-In-Memory Benchmarks), SAFARI Research Group GitHub, [https://github.com/CMU-SAFARI/prim-benchmarks](https://github.com/CMU-SAFARI/prim-benchmarks), accessed 16 July 2024,
- PIM-ML, SAFARI Research Group GitHub, [https://github.com/CMU-SAFARI/pim-ml](https://github.com/CMU-SAFARI/pim-ml),accessed 16 July 2024,
- TransPimLib: A Library for Efficient Transcendental Functions on Processing-in-Memory Systems, SAFARI Research Group GitHub, [https://github.com/CMU-SAFARI/transpimlib](https://github.com/CMU-SAFARI/transpimlib), accessed 16 July 2024,
- RowPress, SAFARI Research Group GitHub, [https://github.com/CMU-SAFARI/RowPress](https://github.com/CMU-SAFARI/RowPress), accessed 16 July 2024, and 
- LambdaML, DS3Lab GitHub, [https://github.com/DS3Lab/LambdaML](https://github.com/DS3Lab/LambdaML), accessed 16 July 2024.


## Getting Help
If you have any suggestions for improvement, please contact steverhyner7 at gmail dot com. If you find any bugs or have further questions or requests, please post an issue at the issue page.

## Acknowledgements
We thank the anonymous reviewers of PACT 2024 for feedback. We thank the SAFARI Research Group members for providing a stimulating intellectual environment. We thank UPMEM for providing hardware resources to perform this research. We acknowledge the generous gifts from our industrial partners, including Google, Huawei, Intel, and Microsoft. This work is supported in part by the Semiconductor Research Corporation (SRC), the ETH Future Computing Laboratory (EFCL), the European Union’s Horizon programme for research and innovation [101047160 - BioPIM], and the AI Chip Center for Emerging Smart Systems, sponsored by InnoHK funding, Hong Kong SAR (ACCESS).