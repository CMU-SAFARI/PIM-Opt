#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>

#include <barrier.h>
#include <seqread.h>


#include "../common_support/common.h"
#include "../common_support/cyclecount.h"

#include "../minimal/_tpl_only_ldexpf_exp/dpu/lut_exp.h"

__host dpu_result_t BENCHMARK[NR_TASKLETS];

__host dpu_arguments_t DIA; 


__mram_noinit LS DPU_RESULTS[N_FEATURES]; 
__dma_aligned LS dot_product_frac_tmp[NR_TASKLETS]; 

__dma_aligned uint32_t mram_base_addr_Y;
__dma_aligned U cache_Y[128]; 
__dma_aligned LS diff_tmp;
__dma_aligned LS label;
__dma_aligned LS bias;
__dma_aligned uint32_t index_Y;


// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// main
int main() {

    unsigned int tasklet_id = me();

    if (tasklet_id == 0) {
        mem_reset(); // Reset the heap

        if (DIA.current_batch_id == DIA.nr_batches) {
            DIA.current_batch_id = 0;
        }
        perfcounter_config(COUNT_CYCLES, true);
    }
    
    perfcounter_cycles_t cycles;
    barrier_wait(&my_barrier); // Barrier
    
    
    dpu_result_t *result = &BENCHMARK[tasklet_id];
    result->cycles_init = (uint64_t) (0);
    result->cycles_compute = (uint64_t) (0);
    result->cycles_mram_to_wram = (uint64_t) (0);
    result->bytes_mram_to_wram = (uint64_t) (0);
    result->cycles_wram_to_mram = (uint64_t) (0);
    result->bytes_wram_to_mram = (uint64_t) (0);

    barrier_wait(&my_barrier); // Barrier
    timer_start(&cycles);

    // for updating the weights
    LS overflow_test = (LS) CAST;
    LS underflow_test = -overflow_test;
    S highest = (S) (overflow_test-1);
    S lowest = (S) (underflow_test+1);


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
    
    uint32_t current_batch_id = DIA.current_batch_id; 
    uint32_t learning_rate = DIA.learning_rate;
    uint32_t reg_term = DIA.reg_term;
    uint32_t learning_rate_plus_reg_term = DIA.learning_rate_plus_reg_term;
    uint32_t task_epochs = DIA.task_epochs;


    uint32_t *tmp_DPU_RESULTS = (uint32_t *) (&DPU_RESULTS);
    U *cache_X = (U *) mem_alloc(16);
    S *tasklet_tmp_cache_W = (S *) mem_alloc(8);
    S *cache_W = (S *) mem_alloc(512);
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
    
    uint32_t mram_base_update_W_addr = (uint32_t) (mram_base_addr_W + tasklet_id * 250368);
    
    uint32_t mram_temp_update_W_addr = mram_base_update_W_addr;

    uint32_t mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
    
    
    result->cycles_init = timer_stop(&cycles);
    barrier_wait(&my_barrier);

    
    
    barrier_wait(&my_barrier);
    timer_start(&cycles);

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
    
    result->cycles_compute += timer_stop(&cycles);

    for (uint32_t i = 0; i < 489; ++i) { // 1000032 + 1440 = 1001472 and 1001472 / 512 = 489
        
        barrier_wait(&my_barrier);
        timer_start(&cycles);

        mram_write((void *) cache_dpu_results,(__mram_ptr void*) (mram_temp_addr_dpu_results), 1024);
        
        barrier_wait(&my_barrier);
        result->cycles_wram_to_mram += timer_stop(&cycles);
        result->bytes_wram_to_mram += (uint64_t) (1024);
        
        barrier_wait(&my_barrier);
        timer_start(&cycles);

        mram_temp_addr_dpu_results += 1024;
        
        result->cycles_compute += timer_stop(&cycles);
    }

    mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
    barrier_wait(&my_barrier);
    

    for (uint32_t batch_size_fraction_id = 0; batch_size_fraction_id < b_size_frac; ++batch_size_fraction_id) {
        barrier_wait(&my_barrier);
        timer_start(&cycles); 

        if (batch_size_fraction_id != 0) {
            mram_temp_addr_X += n_size_samples_byte;
        }

        
        result->cycles_compute += timer_stop(&cycles);
                
        barrier_wait(&my_barrier);
        timer_start(&cycles);
        
        mram_read((__mram_ptr void const*) (mram_temp_addr_X), cache_X, number_of_bytes_to_read_X);
        
        barrier_wait(&my_barrier);
        result->cycles_mram_to_wram += timer_stop(&cycles);
        result->bytes_mram_to_wram += (uint64_t) (number_of_bytes_to_read_X);

        barrier_wait(&my_barrier);
        timer_start(&cycles); 
        
        if (tasklet_id == 0) {
            
            // allocate CACHE_Y by tasklet 0 
            if (batch_size_fraction_id % 128 == 0) {
                index_Y = 0;
                
                result->cycles_compute += timer_stop(&cycles);
                timer_start(&cycles);
                mram_read((__mram_ptr void const*) (mram_temp_addr_Y), (void*) cache_Y, 512); 
                
                result->cycles_mram_to_wram += timer_stop(&cycles);
                result->bytes_mram_to_wram += (uint64_t) (512); 
                timer_start(&cycles);
                mram_temp_addr_Y += 512;

            }
        }

        // Computing dot_product
        dot_product_frac_tmp[tasklet_id] = (LS) (0);
        uint32_t tmp_address = (uint32_t) (0);
        uint32_t current_W_location_tmp = (uint32_t) (0);
        uint32_t current_W_index_tmp = (uint32_t) (0);

        
        result->cycles_compute += timer_stop(&cycles);

        for (uint32_t k = 0; k < number_of_features_to_process; ++k) {
            
            timer_start(&cycles);

            current_W_index_tmp = (uint32_t) (0);
            current_W_location_tmp = (uint32_t) (mram_base_addr_W + (cache_X[k]));
            if (current_W_location_tmp % 8 != 0) {
                current_W_location_tmp -= 4;
                current_W_index_tmp = (uint32_t) (1);
            }
            
            result->cycles_compute += timer_stop(&cycles);
            
            timer_start(&cycles);

            mram_read((__mram_ptr void const*) (current_W_location_tmp), (void *) tasklet_tmp_cache_W, 8); 
            
            result->cycles_mram_to_wram += timer_stop(&cycles);
            result->bytes_mram_to_wram += (uint64_t) (8);

            
            timer_start(&cycles);

            dot_product_frac_tmp[tasklet_id] += (LS) (tasklet_tmp_cache_W[current_W_index_tmp]);
            
            result->cycles_compute += timer_stop(&cycles);

        }


        // tasklet_15 responsible for bias
        
        barrier_wait(&my_barrier);
        timer_start(&cycles);

        if (tasklet_id == 15) {
            if (batch_size_fraction_id == 0) {
                current_W_index_tmp = (uint32_t) (0);
                current_W_location_tmp = (uint32_t) (mram_base_addr_W + 4000000); // 4*1000000, bias stored at location 1000000
                if (current_W_location_tmp % 8 != 0) {
                    current_W_location_tmp -= 4;
                    current_W_index_tmp = (uint32_t) (1);
                }
                
                result->cycles_compute += timer_stop(&cycles);
                timer_start(&cycles);

                mram_read((__mram_ptr void const*) (current_W_location_tmp), (void *) tasklet_tmp_cache_W, 8); 
                
                result->cycles_mram_to_wram += timer_stop(&cycles);
                result->bytes_mram_to_wram += (uint64_t) (8);
                timer_start(&cycles);
                
                bias = (LS) (tasklet_tmp_cache_W[current_W_index_tmp]);
            }
            dot_product_frac_tmp[tasklet_id] += bias;
        }

        result->cycles_compute += timer_stop(&cycles);

        barrier_wait(&my_barrier);
        
        timer_start(&cycles);

        if (tasklet_id == 0) {
            diff_tmp = (LS) (0);

            LS dot_product_tmp = (LS) (0);
            for (uint32_t k = 0; k < NR_TASKLETS; ++k) {
                dot_product_tmp += dot_product_frac_tmp[k];
            }
            
            LS sigmoid = (LS) (0);
            float exp_tmp = expf(-((float) dot_product_tmp / CAST));
            
            sigmoid = (LS)  (CAST / (1.0 + exp_tmp));
            label = cache_Y[index_Y] == 1 ? ((LS) (CAST)) : ((LS) (0));

            diff_tmp = (LS) (sigmoid - label);
            ++index_Y;
        }
        
        result->cycles_compute += timer_stop(&cycles);
        barrier_wait(&my_barrier);

        
        timer_start(&cycles);
        
        LS check_tmp = (LS) (0);
        LS check_tmp_1 = (LS) (0);
        

        // update gradient

        
        tmp_address = (uint32_t) (0);
        uint32_t current_dpu_results_location_tmp = (uint32_t) (0);
        uint32_t current_dpu_results_index_tmp = (uint32_t) (0);
        
        result->cycles_compute += timer_stop(&cycles);

        for (uint32_t k = 0; k < number_of_features_to_process; ++k) {
           
            timer_start(&cycles);
            current_dpu_results_location_tmp = (uint32_t) (*tmp_DPU_RESULTS + ((cache_X[k])<<1));
            
            
            result->cycles_compute += timer_stop(&cycles);
            
            timer_start(&cycles);
            
            mram_read((__mram_ptr void const*) (current_dpu_results_location_tmp), (void *) tasklet_cache_dpu_results, 8);

            
            result->cycles_mram_to_wram += timer_stop(&cycles);
            result->bytes_mram_to_wram += (uint64_t) (8);
            
            timer_start(&cycles); 
            
            check_tmp = (LS) (diff_tmp >> learning_rate);

            if (check_tmp != -1) { 
                if (label == 0) {
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
            
            
            result->cycles_compute += timer_stop(&cycles);
            
            if (tasklet_cache_dpu_results[current_dpu_results_index_tmp] != -1) { 
                if (check_tmp != -1) {
                    
                    timer_start(&cycles);                               
                    tasklet_cache_dpu_results[current_dpu_results_index_tmp] += check_tmp;
                    
                    result->cycles_compute += timer_stop(&cycles);
                    
                    timer_start(&cycles);
                    mram_write((void *) tasklet_cache_dpu_results,(__mram_ptr void*) (current_dpu_results_location_tmp), 8);
                    
                    result->cycles_wram_to_mram += timer_stop(&cycles);
                    result->bytes_wram_to_mram += (uint64_t) (8);
                }
            }
            
            
        }


        // update bias term
        
        barrier_wait(&my_barrier);
        timer_start(&cycles);
        
        if (tasklet_id == 15) {
            LS check_tmp_bias = (LS) (0);
            LS check_tmp_bias_1 = (LS) (0);
            uint32_t current_dpu_results_location_tmp_bias = (uint32_t) (0);
            uint32_t current_dpu_results_index_tmp_bias = (uint32_t) (0);
            
            current_dpu_results_location_tmp_bias = (uint32_t) (*tmp_DPU_RESULTS + 8000000);//(4000000<<1));
            
            result->cycles_compute += timer_stop(&cycles);
            timer_start(&cycles);
            
            mram_read((__mram_ptr void const*) (current_dpu_results_location_tmp_bias), (void *) tasklet_cache_dpu_results, 8); 
            
            result->cycles_mram_to_wram += timer_stop(&cycles);
            result->bytes_mram_to_wram += (uint64_t) (8);
            timer_start(&cycles);
            
            check_tmp_bias = (LS) (diff_tmp >> learning_rate);
            if (check_tmp_bias != -1) { 
                if (label == 0) {
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
                tasklet_cache_dpu_results[current_dpu_results_index_tmp_bias] += check_tmp_bias;
            }
            
            result->cycles_compute += timer_stop(&cycles);
            timer_start(&cycles);
            mram_write((void *) tasklet_cache_dpu_results,(__mram_ptr void*) (current_dpu_results_location_tmp_bias), 8);
            
            result->cycles_wram_to_mram += timer_stop(&cycles);
            result->bytes_wram_to_mram += (uint64_t) (8);
            timer_start(&cycles);

            
        } 
        
        result->cycles_compute += timer_stop(&cycles);
        
        
        
    } // iteration over all elements of b_size_frac done
    
    // Update model

    // tasklet 15 is responsible for updating
    barrier_wait(&my_barrier);

    
    timer_start(&cycles);
        
    LS tmp = (LS) (0);
    LS gradient_update = (LS) (0);
    LS check_tmp_reg = (LS) (0);

    
    result->cycles_compute += timer_stop(&cycles);

    if (tasklet_id < 16) { 
        for (uint32_t i = 0; i < 489; ++i) { // 1000032 + 1440 = 1001472 and 1001472 / 512 = 489
            
            barrier_wait(&my_barrier);
            timer_start(&cycles);
            
            mram_read((__mram_ptr void const*) (mram_temp_update_W_addr), (void *) cache_W, 512);
            mram_read((__mram_ptr void*) (mram_temp_addr_dpu_results), (void *) cache_dpu_results, 1024);
            
            barrier_wait(&my_barrier);
            result->cycles_mram_to_wram += timer_stop(&cycles);
            result->bytes_mram_to_wram += (uint64_t) (512 + 1024);
            barrier_wait(&my_barrier);
            timer_start(&cycles); 


            for (uint32_t k = 0; k < 128; ++k) { // 512 /4 = 128
                gradient_update = (LS) (0);
                
                cache_dpu_results[k] = (LS) (cache_dpu_results[k] >> b_size_frac_log);
                check_tmp_reg = (LS) (((LS) (cache_W[k])) >> (learning_rate_plus_reg_term));

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
            
            result->cycles_compute += timer_stop(&cycles);
            barrier_wait(&my_barrier);
            timer_start(&cycles);
            mram_write((void *) cache_W,(__mram_ptr void*) (mram_temp_update_W_addr), 512);
            
            barrier_wait(&my_barrier);
            result->cycles_wram_to_mram += timer_stop(&cycles);
            result->bytes_wram_to_mram += (uint64_t) (512);

            barrier_wait(&my_barrier);
            timer_start(&cycles);

            mram_temp_update_W_addr += 512;
            mram_temp_addr_dpu_results += 1024;

            
            result->cycles_compute += timer_stop(&cycles);
        }
    } else {
        for (uint32_t i = 0; i < 477; ++i) { // 1000032 + 1440 = 1001472 and 1001472 / 512 = 489
             
            barrier_wait(&my_barrier);
            timer_start(&cycles);

            mram_read((__mram_ptr void const*) (mram_temp_update_W_addr), (void *) cache_W, 512);
            mram_read((__mram_ptr void*) (mram_temp_addr_dpu_results), (void *) cache_dpu_results, 1024);
            
            barrier_wait(&my_barrier);
            result->cycles_mram_to_wram += timer_stop(&cycles);
            result->bytes_mram_to_wram += (uint64_t) (512 + 1024);
            barrier_wait(&my_barrier);
            timer_start(&cycles); 


            for (uint32_t k = 0; k < 128; ++k) { // 512 /4 = 128
                gradient_update = (LS) (0);
                
                cache_dpu_results[k] = (LS) (cache_dpu_results[k] >> b_size_frac_log);
                check_tmp_reg = (LS) (((LS) (cache_W[k])) >> (learning_rate_plus_reg_term));

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
            
            result->cycles_compute += timer_stop(&cycles);
            barrier_wait(&my_barrier);
            timer_start(&cycles);

            mram_write((void *) cache_W,(__mram_ptr void*) (mram_temp_update_W_addr), 512);
            
            barrier_wait(&my_barrier);
            result->cycles_wram_to_mram += timer_stop(&cycles);
            result->bytes_wram_to_mram += (uint64_t) (512);

            barrier_wait(&my_barrier);
            timer_start(&cycles);

            mram_temp_update_W_addr += 512;
            mram_temp_addr_dpu_results += 1024;
            
            result->cycles_compute += timer_stop(&cycles);
        }

        
        barrier_wait(&my_barrier);
        timer_start(&cycles);

        mram_read((__mram_ptr void const*) (mram_temp_update_W_addr), (void *) cache_W, 512);
        mram_read((__mram_ptr void*) (mram_temp_addr_dpu_results), (void *) cache_dpu_results, 1024);
        
        barrier_wait(&my_barrier);
        result->cycles_mram_to_wram += timer_stop(&cycles);
        result->bytes_mram_to_wram += (uint64_t) (512 + 1024);
        barrier_wait(&my_barrier);
        timer_start(&cycles); 


        for (uint32_t k = 0; k < 64; ++k) { // 512 /4 = 128
            gradient_update = (LS) (0);
            cache_dpu_results[k] = (LS) (cache_dpu_results[k] >> b_size_frac_log);
            check_tmp_reg = (LS) (((LS) (cache_W[k])) >> (learning_rate_plus_reg_term));

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
        
        result->cycles_compute += timer_stop(&cycles);
        barrier_wait(&my_barrier);
        timer_start(&cycles);

        mram_write((void *) cache_W,(__mram_ptr void*) (mram_temp_update_W_addr), 512);
        
        barrier_wait(&my_barrier);
        result->cycles_wram_to_mram += timer_stop(&cycles);
        result->bytes_wram_to_mram += (uint64_t) (512);

        barrier_wait(&my_barrier);
        timer_start(&cycles);

        mram_temp_update_W_addr += 512;
        mram_temp_addr_dpu_results += 1024;
        
        result->cycles_compute += timer_stop(&cycles);
    }
    


    barrier_wait(&my_barrier);
    
    timer_start(&cycles);
    
    mram_temp_update_W_addr = mram_base_update_W_addr;
    mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
    

    
    result->cycles_compute += timer_stop(&cycles);

    
    if (tasklet_id == 0) {
        DIA.current_batch_id = DIA.current_batch_id + 1;
    }

    
    // Barrier
    barrier_wait(&my_barrier);

    return 0;
}
