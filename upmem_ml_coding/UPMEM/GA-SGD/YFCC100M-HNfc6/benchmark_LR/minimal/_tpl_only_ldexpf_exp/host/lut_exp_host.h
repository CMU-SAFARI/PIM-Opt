#ifndef LUT_EXP_HOST
#define LUT_EXP_HOST

#include <assert.h>
#include <dpu.h>
#include <math.h>

#define EXP_PRECISION 20

void broadcast_tables(struct dpu_set_t set);

#endif
