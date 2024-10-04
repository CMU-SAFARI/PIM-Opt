#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>


#include "../common_support/common.h"


__host dpu_arguments_t DIA;


__mram_noinit LS DPU_RESULTS[N_FEATURES]; 
__dma_aligned LS dot_product_frac_tmp[NR_TASKLETS]; 

__dma_aligned uint32_t mram_base_addr_Y;
__dma_aligned U cache_Y[128]; 
__dma_aligned LS label;
__dma_aligned LS bias;
__dma_aligned U check_smaller_one;
__dma_aligned uint32_t index_Y;


// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// main
int main() {

    unsigned int tasklet_id = me();

    if (tasklet_id == 0) {
        mem_reset(); // Reset the heap
    }
    barrier_wait(&my_barrier); // Barrier

    // for updating the weights
    LS overflow_test = (LS) CAST;
    LS underflow_test = -overflow_test;
    S highest = (S) (overflow_test-1);
    S lowest = (S) (underflow_test+1);

    LS one = (LS) (CAST);
    LS minus_one = -one;

    LS one_half = (LS) (CONSTRAINT);

    uint32_t n_size = DIA.n_size;
    uint32_t n_size_pad = DIA.n_size_pad;
    uint32_t n_size_samples = DIA.n_size_samples;


    uint32_t batch_size_frac_Y_pad = DIA.batch_size_frac_Y_pad;
    uint32_t communicate_Y_pad = DIA.communicate_Y_pad;

    uint32_t max_rows = DIA.max_rows;

    // For iterating over samples of batch, i.e. iterating over batch_size_fraction_id
    uint32_t n_size_byte = n_size << MUL_U;//* sizeof(U);
    uint32_t n_size_samples_byte = n_size_samples << MUL_U;
    uint32_t n_size_pad_byte = n_size_pad << MUL_U;//* sizeof(U);

    uint32_t b_size_frac = DIA.b_size_frac;
    uint32_t b_size_frac_log = DIA.b_size_frac_log;
    uint32_t batch_size_frac_byte_Y = b_size_frac << MUL_U;
    uint32_t batch_size_frac_pad_byte_Y = batch_size_frac_Y_pad << MUL_U;


    uint32_t communicate_Y_pad_byte = communicate_Y_pad << MUL_U;

    uint32_t nr_batches = DIA.nr_batches;
    
    uint32_t current_batch_id = (U) (0);
    uint32_t learning_rate = DIA.learning_rate;
    uint32_t reg_term = DIA.reg_term;
    uint32_t learning_rate_plus_reg_term = DIA.learning_rate_plus_reg_term;
    uint32_t task_epochs = DIA.task_epochs;


    uint32_t *tmp_DPU_RESULTS = (uint32_t *) (&DPU_RESULTS);
    U *cache_X = (U *) mem_alloc(16);
    S *tasklet_tmp_cache_W = (S *) mem_alloc(8);
    S *cache_W = (S *) mem_alloc(512);
    S *cache_U_Z = (S *) mem_alloc(512);
    LS *cache_dpu_results = (LS *) mem_alloc(1024);
    LS *tasklet_cache_dpu_results = (LS *) mem_alloc(8); 

    
    uint32_t start_index = (uint32_t) (0);
    uint32_t start_index_byte = (uint32_t) (0);
    uint32_t number_of_bytes_to_read_X = (uint32_t) (0);
    uint32_t number_of_features_to_process = (uint32_t) (0);
    if (tasklet_id < 7) {
        start_index = (uint32_t) (tasklet_id * 3);
        start_index_byte = (uint32_t) (tasklet_id * (3 << MUL_U));
        number_of_bytes_to_read_X = (uint32_t) (16);
        number_of_features_to_process = (uint32_t) (3);
    } else {
        start_index = (uint32_t) (21 + ((tasklet_id-7) << 1)); 
        start_index_byte = (uint32_t) (7 * (3 << MUL_U));
        start_index_byte += (uint32_t) ((tasklet_id-7) * (2 << MUL_U));
        number_of_bytes_to_read_X = (uint32_t) (8);
        number_of_features_to_process = (uint32_t) (2);
    }
    uint32_t mram_offset = max_rows * n_size_samples_byte;
    uint32_t mram_batch_offset = b_size_frac*n_size_samples_byte;
    
    uint32_t mram_base_addr_W = (uint32_t) (DPU_MRAM_HEAP_POINTER + mram_offset + communicate_Y_pad_byte);
    uint32_t mram_base_addr_dpu_results = (uint32_t) ((*tmp_DPU_RESULTS) + tasklet_id * (250368<<1)); // (4*1001472)/16
    uint32_t mram_base_addr_U_Z = (uint32_t) (DPU_MRAM_HEAP_POINTER + mram_offset + communicate_Y_pad_byte + n_size_pad_byte);
    
    uint32_t mram_base_update_W_addr = (uint32_t) (mram_base_addr_W + tasklet_id * 250368);
    uint32_t mram_base_update_U_Z_addr = (uint32_t) (mram_base_addr_U_Z + tasklet_id * 250368);
    
    uint32_t mram_temp_update_W_addr = mram_base_update_W_addr;
    uint32_t mram_temp_update_U_Z_addr = mram_base_update_U_Z_addr;

    uint32_t mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
    
    
    for (uint32_t rep = 0; rep < task_epochs; ++rep) {
        current_batch_id = (U) (0);

        while (current_batch_id < nr_batches) { 
            uint32_t mram_base_addr_X = (uint32_t) (DPU_MRAM_HEAP_POINTER + current_batch_id*mram_batch_offset+ start_index_byte); 

            if (tasklet_id == 0) {
               mram_base_addr_Y = (uint32_t) (DPU_MRAM_HEAP_POINTER + mram_offset + current_batch_id*batch_size_frac_byte_Y); //the labels that are relevant for this tasklet
            }

            
            uint32_t mram_temp_addr_X = mram_base_addr_X;
            uint32_t mram_temp_addr_Y = mram_base_addr_Y;
            mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
            mram_temp_update_W_addr = mram_base_update_W_addr;

            


            for (uint32_t i = 0; i < 128; ++i) {
                cache_dpu_results[i] = (LS) (0);
            }
            for (uint32_t i = 0; i < 489; ++i) { // 1000032 + 1440 = 1001472 and 1001472 / 512 = 489
                mram_write((void *) cache_dpu_results,(__mram_ptr void*) (mram_temp_addr_dpu_results), 1024);
                mram_temp_addr_dpu_results += 1024;
            }

            mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
            barrier_wait(&my_barrier);
            

            for (uint32_t batch_size_fraction_id = 0; batch_size_fraction_id < b_size_frac; ++batch_size_fraction_id) {
                 

                if (batch_size_fraction_id != 0) {
                    mram_temp_addr_X += n_size_samples_byte;
                }
                
                mram_read((__mram_ptr void const*) (mram_temp_addr_X), cache_X, number_of_bytes_to_read_X); 
                
                if (tasklet_id == 0) {
                    
                    // allocate CACHE_Y by tasklet 0 
                    if (batch_size_fraction_id % 128 == 0) {
                        index_Y = 0;
                        mram_read((__mram_ptr void const*) (mram_temp_addr_Y), (void*) cache_Y, 512); 
                        mram_temp_addr_Y += 512;

                    }
                }

                // Computing dot_product
                dot_product_frac_tmp[tasklet_id] = (LS) (0);
                uint32_t tmp_address = (uint32_t) (0);
                uint32_t current_W_location_tmp = (uint32_t) (0);
                uint32_t current_W_index_tmp = (uint32_t) (0);
                for (uint32_t k = 0; k < number_of_features_to_process; ++k) {
                    current_W_index_tmp = (uint32_t) (0);
                    current_W_location_tmp = (uint32_t) (mram_base_addr_W + (cache_X[k]));
                    if (current_W_location_tmp % 8 != 0) {
                        current_W_location_tmp -= 4;
                        current_W_index_tmp = (uint32_t) (1);
                    }
                    mram_read((__mram_ptr void const*) (current_W_location_tmp), (void *) tasklet_tmp_cache_W, 8); 
                    dot_product_frac_tmp[tasklet_id] += (LS) (tasklet_tmp_cache_W[current_W_index_tmp]);

                }


                // tasklet_15 responsible for bias
                if (tasklet_id == 15) {
                    if (batch_size_fraction_id == 0) {
                        current_W_index_tmp = (uint32_t) (0);
                        current_W_location_tmp = (uint32_t) (mram_base_addr_W + 4000000); // 4*1000000, bias stored at location 1000000
                        if (current_W_location_tmp % 8 != 0) {
                            current_W_location_tmp -= 4;
                            current_W_index_tmp = (uint32_t) (1);
                        }
                        mram_read((__mram_ptr void const*) (current_W_location_tmp), (void *) tasklet_tmp_cache_W, 8); 
                        bias = (LS) (tasklet_tmp_cache_W[current_W_index_tmp]);
                    }
                    dot_product_frac_tmp[tasklet_id] += bias;
                }

                barrier_wait(&my_barrier);
    
                if (tasklet_id == 0) {
                    

                    LS dot_product_tmp = (LS) (0);
                    for (uint32_t k = 0; k < NR_TASKLETS; ++k) {
                        dot_product_tmp += dot_product_frac_tmp[k];
                    } 
                    label = cache_Y[index_Y] == 1 ? one : minus_one;
                    if (label == minus_one) {
                        dot_product_tmp = (~dot_product_tmp) + 1;

                    }
                    
                    check_smaller_one = (U) (0); // false
                    if (dot_product_tmp < one_half) {
                        check_smaller_one = (U) (1);
                    }

                    ++index_Y;
                }
                barrier_wait(&my_barrier);
                
                LS check_tmp = (LS) (0);
                LS check_tmp_1 = (LS) (0);

                // update gradient
                tmp_address = (uint32_t) (0);
                uint32_t current_dpu_results_location_tmp = (uint32_t) (0);
                uint32_t current_dpu_results_index_tmp = (uint32_t) (0);
                for (uint32_t k = 0; k < number_of_features_to_process; ++k) {
                    current_dpu_results_location_tmp = (uint32_t) (*tmp_DPU_RESULTS + ((cache_X[k])<<1));
                    
                    mram_read((__mram_ptr void const*) (current_dpu_results_location_tmp), (void *) tasklet_cache_dpu_results, 8); 
                    
                    check_tmp = (LS) (CAST >> learning_rate);

                    if (check_tmp != -1) { 
                        if (label == minus_one) { 
                            check_tmp_1 = check_tmp >> 1;
                            if (check_tmp_1 != -1) {
                                check_tmp = check_tmp_1;
                            } 
                        } else {
                            check_tmp_1 = (check_tmp << 4) - check_tmp;
                            if (check_tmp_1 != -1) {
                                check_tmp = check_tmp_1;
                            } 
                        }
                    } 
                    
                    
                    
                    if (tasklet_cache_dpu_results[current_dpu_results_index_tmp] != -1) { 
                        if (check_smaller_one != 0) {
                            if (check_tmp != -1) {
                                if (label == one) {
                                    tasklet_cache_dpu_results[current_dpu_results_index_tmp] -= check_tmp;
                                } else {
                                    tasklet_cache_dpu_results[current_dpu_results_index_tmp] += check_tmp;
                                }
                            }
                            mram_write((void *) tasklet_cache_dpu_results,(__mram_ptr void*) (current_dpu_results_location_tmp), 8);
                        }   
                        
                    }
                    
                    
                }


                // update bias term
                
                if (tasklet_id == 15) {
                    LS check_tmp_bias = (LS) (0);
                    LS check_tmp_bias_1 = (LS) (0);
                    uint32_t current_dpu_results_location_tmp_bias = (uint32_t) (0);
                    uint32_t current_dpu_results_index_tmp_bias = (uint32_t) (0);
                    
                    current_dpu_results_location_tmp_bias = (uint32_t) (*tmp_DPU_RESULTS + 8000000);//(4000000<<1));
                    
                    mram_read((__mram_ptr void const*) (current_dpu_results_location_tmp_bias), (void *) tasklet_cache_dpu_results, 8); 
                    
                    check_tmp_bias = (LS) (CAST >> learning_rate);
                    if (check_tmp_bias != -1) { 
                        if (label == minus_one) {
                            check_tmp_bias_1 = check_tmp_bias >> 1;
                            if (check_tmp_bias_1 != -1) {
                                check_tmp_bias = check_tmp_bias_1;
                            } 
                        } else {
                            check_tmp_bias_1 = (check_tmp_bias << 4) - check_tmp_bias;
                            if (check_tmp_bias_1 != -1) {
                                check_tmp_bias = check_tmp_bias_1;
                            } 
                        }
                    } 
                    
                    if (check_tmp_bias != -1) {
                        if (check_smaller_one != 0) {
                            if (label == one) {
                                tasklet_cache_dpu_results[current_dpu_results_index_tmp_bias] -= check_tmp_bias;
                            } else {
                                tasklet_cache_dpu_results[current_dpu_results_index_tmp_bias] += check_tmp_bias;
                            }
                        }
                        
                    }
                    mram_write((void *) tasklet_cache_dpu_results,(__mram_ptr void*) (current_dpu_results_location_tmp_bias), 8);

                    
                } 
                
                
                
            } // iteration over all elements of b_size_frac done
            
            // Update model

            // tasklet 15 is responsible for updating
            barrier_wait(&my_barrier);
             
            LS tmp = (LS) (0);
            LS gradient_update = (LS) (0);
            LS check_tmp_reg = (LS) (0);

            if (tasklet_id < 15) { 
                for (uint32_t i = 0; i < 489; ++i) { // 1000032 + 1440 = 1001472 and 1001472 / 512 = 489
                    mram_read((__mram_ptr void const*) (mram_temp_update_U_Z_addr), (void *) cache_U_Z, 512);
                    mram_read((__mram_ptr void const*) (mram_temp_update_W_addr), (void *) cache_W, 512);
                    mram_read((__mram_ptr void*) (mram_temp_addr_dpu_results), (void *) cache_dpu_results, 1024);


                    for (uint32_t k = 0; k < 128; ++k) { // 512 /4 = 128
                        gradient_update = (LS) (0);
                        cache_dpu_results[k] = (LS) (cache_dpu_results[k] >> b_size_frac_log);
                        check_tmp_reg = (LS) ((((LS) (cache_W[k])) + ((LS) (cache_U_Z[k])) )>> (learning_rate_plus_reg_term));
                        
                        if (cache_dpu_results[k] != -1) {
                            if (check_tmp_reg != -1) {
                                gradient_update = (LS) (cache_dpu_results[k] + check_tmp_reg);
                            } else {
                                gradient_update = cache_dpu_results[k];
                            }
                        } else {
                            if (check_tmp_reg != -1) {
                                gradient_update = check_tmp_reg;
                            }
                        }

                        
                        tmp = (LS) (((LS) cache_W[k]) - gradient_update);
                        if (tmp > overflow_test || tmp < underflow_test) {
                            if (tmp > overflow_test) {
                                cache_W[k] = highest; 
                            } else {
                                cache_W[k] = lowest;
                            }
                        } else {
                            cache_W[k] = (S) tmp;
                        }
                    }
                    mram_write((void *) cache_W,(__mram_ptr void*) (mram_temp_update_W_addr), 512);
                    mram_temp_update_U_Z_addr += 512;
                    mram_temp_update_W_addr += 512;
                    mram_temp_addr_dpu_results += 1024;
                }
            } else {
                for (uint32_t i = 0; i < 477; ++i) { // 1000032 + 1440 = 1001472 and 1001472 / 512 = 489
                    
                    mram_read((__mram_ptr void const*) (mram_temp_update_U_Z_addr), (void *) cache_U_Z, 512);
                    mram_read((__mram_ptr void const*) (mram_temp_update_W_addr), (void *) cache_W, 512);
                    mram_read((__mram_ptr void*) (mram_temp_addr_dpu_results), (void *) cache_dpu_results, 1024);


                    for (uint32_t k = 0; k < 128; ++k) { // 512 /4 = 128
                        gradient_update = (LS) (0);
                        cache_dpu_results[k] = (LS) (cache_dpu_results[k] >> b_size_frac_log);
                        check_tmp_reg = (LS) ((((LS) (cache_W[k])) + ((LS) (cache_U_Z[k])) )>> (learning_rate_plus_reg_term));
                        
                        if (cache_dpu_results[k] != -1) {
                            if (check_tmp_reg != -1) {
                                gradient_update = (LS) (cache_dpu_results[k] + check_tmp_reg);
                            } else {
                                gradient_update = cache_dpu_results[k];
                            }
                        } else {
                            if (check_tmp_reg != -1) {
                                gradient_update = check_tmp_reg;
                            }
                        }

                        
                        tmp = (LS) (((LS) cache_W[k]) - gradient_update);
                        if (tmp > overflow_test || tmp < underflow_test) {
                            if (tmp > overflow_test) {
                                cache_W[k] = highest; 
                            } else {
                                cache_W[k] = lowest;
                            }
                        } else {
                            cache_W[k] = (S) tmp;
                        }
                    }
                    mram_write((void *) cache_W,(__mram_ptr void*) (mram_temp_update_W_addr), 512);
                    mram_temp_update_U_Z_addr += 512;
                    mram_temp_update_W_addr += 512;
                    mram_temp_addr_dpu_results += 1024;
                }
                mram_read((__mram_ptr void const*) (mram_temp_update_U_Z_addr), (void *) cache_U_Z, 512);
                mram_read((__mram_ptr void const*) (mram_temp_update_W_addr), (void *) cache_W, 512);
                mram_read((__mram_ptr void*) (mram_temp_addr_dpu_results), (void *) cache_dpu_results, 1024);


                for (uint32_t k = 0; k < 64; ++k) { // 512 /4 = 128
                    gradient_update = (LS) (0);
                    cache_dpu_results[k] = (LS) (cache_dpu_results[k] >> b_size_frac_log);
                    check_tmp_reg = (LS) ((((LS) (cache_W[k])) + ((LS) (cache_U_Z[k])) )>> (learning_rate_plus_reg_term));
                    
                    if (cache_dpu_results[k] != -1) {
                        if (check_tmp_reg != -1) {
                            gradient_update = (LS) (cache_dpu_results[k] + check_tmp_reg);
                        } else {
                            gradient_update = cache_dpu_results[k];
                        }
                    } else {
                        if (check_tmp_reg != -1) {
                            gradient_update = check_tmp_reg;
                        }
                    }

                    
                    tmp = (LS) (((LS) cache_W[k]) - gradient_update);
                    if (tmp > overflow_test || tmp < underflow_test) {
                        if (tmp > overflow_test) {
                            cache_W[k] = highest; 
                        } else {
                            cache_W[k] = lowest;
                        }
                    } else {
                        cache_W[k] = (S) tmp;
                    }
                }
                // update bias term without regularization
                gradient_update = (LS) (0);
                cache_dpu_results[64] = (LS) (cache_dpu_results[64] >> b_size_frac_log);
                
                
                if (cache_dpu_results[64] != -1) {
                    gradient_update = cache_dpu_results[64];
                } 

                
                tmp = (LS) (((LS) cache_W[64]) - gradient_update);
                if (tmp > overflow_test || tmp < underflow_test) {
                    if (tmp > overflow_test) {
                        cache_W[64] = highest; 
                    } else {
                        cache_W[64] = lowest;
                    }
                } else {
                    cache_W[64] = (S) tmp;
                }



                mram_write((void *) cache_W,(__mram_ptr void*) (mram_temp_update_W_addr), 512);
                mram_temp_update_W_addr += 512; 
                mram_temp_addr_dpu_results += 1024; 
            }
            

            barrier_wait(&my_barrier);
            
            mram_temp_update_U_Z_addr = mram_base_update_U_Z_addr;
            mram_temp_update_W_addr = mram_base_update_W_addr;
            mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
            
            ++current_batch_id;
        }
    }
    
    // Barrier
    barrier_wait(&my_barrier);

    return 0;
}
