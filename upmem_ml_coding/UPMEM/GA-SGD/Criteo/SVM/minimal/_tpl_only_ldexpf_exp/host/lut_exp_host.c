#include "lut_exp_host.h"

int _unused_zero_address; // For some LUTs it is not needed to save the zero address (because it is zero anyway, in that case, we just point to this value)

/*
 * Fills a table on the host side and makes it ready to transmit
 * Inputs
 * xLower: Lower end of the table
 * xUpper: Upper end of the table
 * original(): original function that should be tabularized
 *
 * Outputs
 * size: array size (in entries)
 * table[]: Array to use for the table
 * zero_address & granularity_exponent: values that the DPU uses to correctly assign inputs to table addresses
 */
void fill_table(float xLower, float xUpper, double (*original)(), int size, float table[], int *zero_address, int *granularity_exponent) {
    float x_granularity = (xUpper - xLower) / (float) size;

    frexpf(x_granularity, granularity_exponent);

    // From that exponent figure out the best possible spacing in x
    float x_granularity_rounded = ldexpf(1.0f, *granularity_exponent);
    *zero_address = (int) (-xLower / (xUpper - xLower) * size);

    for(int i = 0; i<size; ++i){
        table[i] = (float) original(x_granularity_rounded * (i - *zero_address));
    }
}

// Generates and Broadcasts all tables to the DPU
void broadcast_tables(struct dpu_set_t set) {
    /***********************************************************
    *   EXP
    */
    float exp_table[1 << EXP_PRECISION];
    int exp_granularity_exponent;

    fill_table(0, log(2), exp, 1 << EXP_PRECISION, exp_table, &_unused_zero_address, &exp_granularity_exponent);
    DPU_ASSERT(dpu_broadcast_to(set, "exp_table", 0, &exp_table, sizeof(exp_table), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(set, "exp_granularity_exponent", 0, &exp_granularity_exponent, sizeof(exp_granularity_exponent), DPU_XFER_DEFAULT));
}
