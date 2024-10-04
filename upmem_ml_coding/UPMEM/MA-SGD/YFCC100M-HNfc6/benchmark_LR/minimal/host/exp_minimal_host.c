#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include "../_tpl_only_ldexpf_exp/host/lut_exp_host.h"

#ifndef DPU_BINARY
#define DPU_BINARY "bin/dpu/exp_minimal"
#endif


int main(void) {
    struct dpu_set_t set, dpu;

    // Get a DPU and load our kernel on it
    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    // Get some number and move it over to the DPU
    float input = 20;
    DPU_ASSERT(dpu_broadcast_to(set, "number", 0, &input, sizeof(float), DPU_XFER_DEFAULT));

    // Fill the tables on the DPU
    broadcast_tables(set);

    // Run the DPU Kernel
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    // Retrieve the result and print it
    float output;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_copy_from(dpu, "number", 0, &output, sizeof(float)));
        printf("DPU calculated exp(%f) = %f.\n", input, output);
    }

    return 0;
}