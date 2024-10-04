#include <mram.h>
#include "../_tpl_only_ldexpf_exp/dpu/lut_exp.h"

__host float number;

int main(){
    number = expf(number); // <<--- Now you can use transcendntal functions!
    return 0;
}