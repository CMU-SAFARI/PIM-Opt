HOST_FLAGS := $ --std=c11 `dpu-pkg-config --cflags --libs dpu` -lm

.PHONY: exp

exp:
	dpu-upmem-dpurte-clang dpu/exp_minimal.c _tpl_only_ldexpf_exp/dpu/lut_exp.c -o bin/dpu/exp_minimal
	clang host/exp_minimal_host.c _tpl_only_ldexpf_exp/host/lut_exp_host.c -o bin/host/exp_minimal_host ${HOST_FLAGS}