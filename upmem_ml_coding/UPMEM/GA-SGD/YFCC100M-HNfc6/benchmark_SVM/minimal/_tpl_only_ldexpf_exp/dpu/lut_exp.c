#include "_ldexpf.c"
#include "_range_extensions.c"
#include "lut_exp.h"

//#pragma ide diagnostic ignored "UnusedParameter"
//#pragma ide diagnostic ignored "UnusedLocalVariable"
//#pragma ide diagnostic ignored "UnusedValue"

/********************************************************************************
 * Basic Usage of Helper Functions:
 *
 *  unsigned int address = float_to_address_roundup_ldexpf(input, table_exponent);
 *  return table[address];
 */

// Helper Function
static inline unsigned int float_to_address_roundup_ldexpf(float x, int exponent) {
    return (((int) ldexpf(x, - exponent + 1)) + 1) >> 1;
}

// Helper Function
static inline unsigned int fixed_to_address_roundup_ldexpf(int x, int exponent) {
    return ((x >> (FIXED_FRACTION_BITS + exponent - 1)) + 1) >> 1;
}

/***********************************************************
*   EXP
*/

__host int exp_granularity_exponent;

#if EXP_STORE_IN_WRAM > 0
__host float exp_table[1 << EXP_PRECISION];
#else
__mram_noinit float exp_table[1 << EXP_PRECISION];
#endif

// Functions
float expf (float x) {
    #ifdef NOWRAP
        int offset_from_zero = float_to_address_roundup_ldexpf(x, exp_granularity_exponent);
        return exp_table[offset_from_zero];
    #else
        int extra_data;
        int offset_from_zero = float_to_address_roundup_ldexpf(exp_range_extension_in(x, &extra_data), exp_granularity_exponent);
        return exp_range_extension_out(exp_table[offset_from_zero], &extra_data);
    #endif
}