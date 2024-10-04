#ifndef LUT_EXP
#define LUT_EXP

#include <mram.h>

#define EXP_PRECISION 20 // This needs to match on CPU and DPU side!
#define EXP_STORE_IN_WRAM 0

float expf(float x);

#endif
