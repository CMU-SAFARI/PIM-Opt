DPU_DIR := dpu
HOST_DIR := host
COMMON_SUPPORT_DIR := common_support
HOST_UTILS_DIR := host_utils
BUILDDIR ?= bin
NR_TASKLETS ?= 16
BL ?= 10 
NR_DPUS ?= 1024
TYPE ?= UINT32
SCALING ?= 0 
AE ?= 0 
SOURCE_DIR ?= /path/to/data
DEST_DIR ?= /path/to/results

define conf_filename
	${BUILDDIR}/.NR_DPUS_$(1)_NR_TASKLETS_$(2)_BL_$(3)_TYPE_$(4)_SCALING_$(5)_AE_$(6).conf
endef
CONF := $(call conf_filename,${NR_DPUS},${NR_TASKLETS},${BL},${TYPE},${SCALING},${AE})

HOST_TARGET := ${BUILDDIR}/host_code
DPU_TARGET := ${BUILDDIR}/dpu_code

COMMON_INCLUDES := ${COMMON_SUPPORT_DIR} 
HOST_UTILS_INCLUDES := ${HOST_UTILS_DIR}
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.c) $(wildcard ${HOST_UTILS_DIR}/*.c)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra -I${COMMON_INCLUDES}
COMMON_HOST_FLAGS := -Wall -Wextra -I${COMMON_INCLUDES} -I${HOST_UTILS_INCLUDES}
HOST_FLAGS := ${COMMON_HOST_FLAGS} -std=c11 -lm -O3 -lpthread `dpu-pkg-config --cflags --libs dpu` -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS} -DBL=${BL} -D${TYPE} -DSCALING=${SCALING} -DAE=${AE} -DSOURCE_DIR=\"${SOURCE_DIR}\" -DDEST_DIR=\"${DEST_DIR}\" -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function
DPU_FLAGS := ${COMMON_FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS} -DBL=${BL} -D${TYPE} -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function

all: ${HOST_TARGET} ${DPU_TARGET}

${CONF}:
	$(RM) $(call conf_filename,*,*)
	touch ${CONF}

${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES} ${HOST_UTILS_INCLUDES} ${CONF}
	$(CC) ${HOST_SOURCES} minimal/_tpl_only_ldexpf_exp/host/lut_exp_host.c -o $@ ${HOST_FLAGS}


${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_SOURCES} minimal/_tpl_only_ldexpf_exp/dpu/lut_exp.c -o $@ ${DPU_FLAGS}

clean:
	$(RM) -r $(BUILDDIR)

test: all
	./${HOST_TARGET}

