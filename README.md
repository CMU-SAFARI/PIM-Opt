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