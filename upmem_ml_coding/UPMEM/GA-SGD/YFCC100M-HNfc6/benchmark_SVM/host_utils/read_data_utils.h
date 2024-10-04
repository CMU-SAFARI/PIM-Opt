#ifndef _READ_DATA_UTILS_H_
#define _READ_DATA_UTILS_H_

#define MAXCHAR 800000 // declare big enough to hold a line from input 


void read_input_yfcc100m(U *X, U *Y, uint64_t length, uint64_t strong_scaling);

void read_test_input_yfcc100m(U *X, U *Y, uint64_t length, uint64_t strong_scaling);



#endif