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
__mram_noinit S DPU_RESULTS[N_FEATURES]; 

__dma_aligned LS dot_product_frac_tmp[NR_TASKLETS]; 
__dma_aligned uint32_t mram_base_addr_Y;
__dma_aligned U cache_Y[64]; 
__dma_aligned LS diff_tmp;




static LS dot_product(U *bufferX, S *bufferW, uint32_t length, uint32_t init_location) {
    LS result = (LS) (0);
    for (uint32_t i = 0; i < length; i++) {
        result += (LS) ( ( ((LS) bufferX[init_location+i]) * ((LS) (bufferW[i])) ) >> SHIFT_AMOUNT );
    }
    return result;
}

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// main
int main() {
    unsigned int tasklet_id = me();

    if (tasklet_id == 0) { // Initialize once the cycle counter
        mem_reset(); // Reset the heap
        if (DIA.current_batch_id == DIA.nr_batches) {
            DIA.current_batch_id = 0;
        }

        perfcounter_config(COUNT_CYCLES, true);
    }
    perfcounter_cycles_t cycles;
    barrier_wait(&my_barrier);
    
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

    uint32_t samples_loop_transfer = DIA.samples_loop_transfer;
    uint32_t transfer = DIA.transfer;
    uint32_t samples_per_transfer = DIA.samples_per_transfer;

    

    uint32_t n_size = DIA.n_size;
    uint32_t n_size_pad = DIA.n_size_pad;
    uint32_t n_size_pad_W_fp = DIA.n_size_pad_W_fp;
    uint32_t n_size_features_frac = DIA.n_size_features_frac;
    uint32_t n_size_features_frac_pad = DIA.n_size_features_frac_pad;

    uint32_t batch_size_frac_Y_pad = DIA.batch_size_frac_Y_pad;
    uint32_t communicate_Y_pad = DIA.communicate_Y_pad;

    uint32_t max_rows = DIA.max_rows;

    uint32_t start_row = tasklet_id;

    // For iterating over samples of batch, i.e. iterating over batch_size_fraction_id
    uint32_t n_size_byte = n_size << MUL_U;//* sizeof(U);
    uint32_t n_size_pad_byte = n_size_pad << MUL_U;//* sizeof(U);

    uint32_t n_size_pad_W_fp_byte = n_size_pad_W_fp << MUL_S;

    // Address of the current row in MRAM
    uint32_t n_size_features_frac_byte = n_size_features_frac << MUL_U;//* sizeof(U);
    uint32_t n_size_features_frac_pad_byte = n_size_features_frac_pad << MUL_U;//* sizeof(U); 

    // Address of the current row in MRAM
    uint32_t n_size_features_frac_byte_W = n_size_features_frac << MUL_S;
    uint32_t n_size_features_frac_pad_byte_W = n_size_features_frac_pad << MUL_S;

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
    uint32_t shift_plus_learning_rate = learning_rate + SHIFT_AMOUNT;
    uint32_t task_epochs = DIA.task_epochs;


    uint32_t *tmp_DPU_RESULTS = (uint32_t *) (&DPU_RESULTS);
    U *cache_X = (U *) mem_alloc(n_size_features_frac_pad_byte); 
    S *cache_W = (S *) mem_alloc(transfer);
    S *cache_dpu_results = (S *) mem_alloc(transfer); 

    uint32_t start_index_byte = start_row * n_size_features_frac_byte;
    uint32_t mram_offset = max_rows * n_size_pad_byte;
    uint32_t mram_batch_offset = b_size_frac*n_size_byte;

    uint32_t mram_base_addr_bias = (uint32_t) (0);
    LS* bias = (LS*) mem_alloc(8);
    LS* bias_gradient = (LS*) mem_alloc(8);
    if (tasklet_id == 15) {
        mram_base_addr_bias = (uint32_t) (DPU_MRAM_HEAP_POINTER + mram_offset + communicate_Y_pad_byte + n_size_pad_W_fp_byte);
        mram_read((__mram_ptr void const*) (mram_base_addr_bias), (void*) bias, 8);
    }

    result->cycles_init = timer_stop(&cycles);
    barrier_wait(&my_barrier);


 
    barrier_wait(&my_barrier);
    timer_start(&cycles);


    uint32_t mram_base_addr_X = (uint32_t) (DPU_MRAM_HEAP_POINTER + current_batch_id*mram_batch_offset+ start_index_byte); 

    if (tasklet_id == 0) {
        mram_base_addr_Y = (uint32_t) (DPU_MRAM_HEAP_POINTER + mram_offset + current_batch_id*batch_size_frac_byte_Y); //the labels that are relevant for this tasklet
    }

    uint32_t mram_base_addr_W = (uint32_t) (DPU_MRAM_HEAP_POINTER + mram_offset + communicate_Y_pad_byte + start_index_byte);
    
    
    uint32_t mram_base_addr_dpu_results = (uint32_t) (*tmp_DPU_RESULTS) + start_index_byte;
    

    uint32_t mram_temp_addr_X = mram_base_addr_X;
    uint32_t mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
    uint32_t mram_temp_addr_W = mram_base_addr_W;

    result->cycles_compute += timer_stop(&cycles);


    for (uint32_t i = 0; i < samples_loop_transfer; i++) {

        barrier_wait(&my_barrier);
        timer_start(&cycles);

        for (uint32_t k = 0; k < samples_per_transfer; ++k) {
            cache_dpu_results[k] = (S) (0);
        }
        result->cycles_compute += timer_stop(&cycles);
        
        timer_start(&cycles);


        mram_write((void *) cache_dpu_results,(__mram_ptr void*) (mram_temp_addr_dpu_results), transfer);
        barrier_wait(&my_barrier);
        result->cycles_wram_to_mram += timer_stop(&cycles);
        result->bytes_wram_to_mram += (uint64_t) (transfer);
        
        barrier_wait(&my_barrier);
        timer_start(&cycles);

        mram_temp_addr_dpu_results += transfer;
        result->cycles_compute += timer_stop(&cycles);
    }
    barrier_wait(&my_barrier);
    timer_start(&cycles);

    mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
    result->cycles_compute += timer_stop(&cycles);
    
    barrier_wait(&my_barrier);
    timer_start(&cycles);
    
    if (tasklet_id == 0) {
        result->cycles_compute += timer_stop(&cycles);
        timer_start(&cycles);

        mram_read((__mram_ptr void const*) (mram_base_addr_Y), (void*) cache_Y, batch_size_frac_pad_byte_Y); 
        result->cycles_mram_to_wram += timer_stop(&cycles);
        result->bytes_mram_to_wram += (uint64_t) (batch_size_frac_pad_byte_Y);
        timer_start(&cycles);
    }
    
    if (tasklet_id == 15) {
        *bias_gradient = (LS) (0);
    }
    result->cycles_compute += timer_stop(&cycles);

    for (uint32_t batch_size_fraction_id = 0; batch_size_fraction_id < b_size_frac; ++batch_size_fraction_id) {
        barrier_wait(&my_barrier);
        timer_start(&cycles);

        mram_temp_addr_X = mram_base_addr_X + batch_size_fraction_id * n_size_byte; 
        result->cycles_compute += timer_stop(&cycles);
        
        barrier_wait(&my_barrier);
        timer_start(&cycles);
        
        mram_read((__mram_ptr void const*) (mram_temp_addr_X), cache_X, n_size_features_frac_pad_byte); 
        barrier_wait(&my_barrier);
        result->cycles_mram_to_wram += timer_stop(&cycles);
        result->bytes_mram_to_wram += (uint64_t) (n_size_features_frac_pad_byte);
        
        barrier_wait(&my_barrier);
        timer_start(&cycles);

        dot_product_frac_tmp[tasklet_id] = (LS) (0);
        result->cycles_compute += timer_stop(&cycles);

        for (uint32_t i = 0; i < samples_loop_transfer; i++) {
            barrier_wait(&my_barrier);
            timer_start(&cycles);

            mram_read((__mram_ptr void const*) (mram_temp_addr_W), (void *) cache_W, transfer); 
            barrier_wait(&my_barrier);
            result->cycles_mram_to_wram += timer_stop(&cycles);
            result->bytes_mram_to_wram += (uint64_t) (transfer);
            
            barrier_wait(&my_barrier);
            timer_start(&cycles);

            dot_product_frac_tmp[tasklet_id] += dot_product(cache_X, cache_W, samples_per_transfer, i*samples_per_transfer);
            mram_temp_addr_W += transfer;

            result->cycles_compute += timer_stop(&cycles);
        }

        barrier_wait(&my_barrier);
        timer_start(&cycles);

        mram_temp_addr_W = mram_base_addr_W; 

        if (tasklet_id == 15) {
            dot_product_frac_tmp[tasklet_id] += *bias;
        }

        result->cycles_compute += timer_stop(&cycles);

        barrier_wait(&my_barrier);
        timer_start(&cycles);

        if (tasklet_id == 0) { 
            diff_tmp = (S) (0);

            LS dot_product_tmp = (LS) (0);
            for (uint32_t k = 0; k < NR_TASKLETS; ++k) {
                dot_product_tmp += dot_product_frac_tmp[k];
            }
            
            LS sigmoid = (LS) (0);
            float exp_tmp = expf(-((float) dot_product_tmp / CAST));
            
            sigmoid = (LS)  (CAST / (1.0 + exp_tmp));
            LS label = cache_Y[batch_size_fraction_id] == 1 ? ((LS) (CAST)) : ((LS) (0));

            diff_tmp = (LS) (sigmoid - label);
            
        }

        result->cycles_compute += timer_stop(&cycles);

        barrier_wait(&my_barrier);

        timer_start(&cycles);
        
        uint32_t gradient_index_tmp = 0;
        LS feature_tmp = (LS) (0); 
        S check_tmp = (S) (0);

        result->cycles_compute += timer_stop(&cycles);

        for (uint32_t i = 0; i < samples_loop_transfer; ++i) {
            barrier_wait(&my_barrier);
            timer_start(&cycles);

            mram_read((__mram_ptr void*) (mram_temp_addr_dpu_results), (void *) cache_dpu_results, transfer);
            barrier_wait(&my_barrier);
            result->cycles_mram_to_wram += timer_stop(&cycles);
            result->bytes_mram_to_wram += (uint64_t) (transfer);
            
            barrier_wait(&my_barrier);
            timer_start(&cycles);

            for (uint32_t k = 0; k < samples_per_transfer; ++k) {

                feature_tmp = (LS) cache_X[gradient_index_tmp];
                check_tmp = (S) ( (feature_tmp * diff_tmp) >> (shift_plus_learning_rate) );
                if (check_tmp != -1) {
                    cache_dpu_results[k] += check_tmp;
                }
                
                ++gradient_index_tmp;

                
            }
            result->cycles_compute += timer_stop(&cycles);
            barrier_wait(&my_barrier);
            timer_start(&cycles);

            mram_write((void *) cache_dpu_results,(__mram_ptr void*) (mram_temp_addr_dpu_results), transfer);
            barrier_wait(&my_barrier);
            result->cycles_wram_to_mram += timer_stop(&cycles);
            result->bytes_wram_to_mram += (uint64_t) (transfer);

            barrier_wait(&my_barrier);
            timer_start(&cycles);

            mram_temp_addr_dpu_results += transfer;
            result->cycles_compute += timer_stop(&cycles);

        }
        barrier_wait(&my_barrier);
        timer_start(&cycles);
        mram_temp_addr_W = mram_base_addr_W;
        mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
        
        // update bias_gradient
        if (tasklet_id == 15) {
            LS check_tmp_bias = (LS) (diff_tmp >> learning_rate);
            if (check_tmp_bias != -1) {
                *bias_gradient += check_tmp_bias;
            }
            

        }
        result->cycles_compute += timer_stop(&cycles);
    } // iteration over all elements of b_size_frac done

    barrier_wait(&my_barrier);


    timer_start(&cycles);
    LS tmp = (LS) (0);
    LS gradient_update = (LS) (0);
    LS check_tmp_reg = (LS) (0);
    result->cycles_compute += timer_stop(&cycles);
    
    for (uint32_t i = 0; i < samples_loop_transfer; ++i) {
        barrier_wait(&my_barrier);
        timer_start(&cycles);

        mram_read((__mram_ptr void const*) (mram_temp_addr_W), (void *) cache_W, transfer);
        mram_read((__mram_ptr void*) (mram_temp_addr_dpu_results), (void *) cache_dpu_results, transfer); 
        barrier_wait(&my_barrier);
        result->cycles_mram_to_wram += timer_stop(&cycles);
        result->bytes_mram_to_wram += (uint64_t) (2*transfer);

        barrier_wait(&my_barrier);
        timer_start(&cycles);
        for (uint32_t k = 0; k < samples_per_transfer; ++k) { 
            gradient_update = (LS) (0);
            cache_dpu_results[k] = (S) (cache_dpu_results[k] >> b_size_frac_log);
            check_tmp_reg = (LS) (((LS) (cache_W[k])) >> (learning_rate_plus_reg_term));

            if (cache_dpu_results[k] != -1) {
                if (check_tmp_reg != -1) {
                    gradient_update = (LS) (((LS) (cache_dpu_results[k])) + check_tmp_reg);
                } else {
                    gradient_update = (LS) (cache_dpu_results[k]);
                }
            } else {
                if (check_tmp_reg != -1) {
                    gradient_update = check_tmp_reg;
                }
            }


            tmp = (LS) ( ((LS) (cache_W[k])) - gradient_update ); 
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

        mram_write((void *) cache_W,(__mram_ptr void*) (mram_temp_addr_W), transfer);
        barrier_wait(&my_barrier);
        result->cycles_wram_to_mram += timer_stop(&cycles);
        result->bytes_wram_to_mram += (uint64_t) (transfer);
        
        barrier_wait(&my_barrier);
        timer_start(&cycles);

        mram_temp_addr_W += transfer;
        mram_temp_addr_dpu_results += transfer;
        result->cycles_compute += timer_stop(&cycles);
    }
    barrier_wait(&my_barrier);
    timer_start(&cycles);

    mram_temp_addr_W = mram_base_addr_W; 
    mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
        
    if (tasklet_id == 15) {
        gradient_update = (LS) (0);
        LS bias_gradient_tmp = *bias_gradient;
        *bias_gradient = (LS) (bias_gradient_tmp >> b_size_frac_log);
        if (*bias_gradient != -1) {
            gradient_update = *bias_gradient;
        }
        tmp = (LS) (*bias - gradient_update);
        if (tmp > overflow_test || tmp < underflow_test) {
            if (tmp > overflow_test) {
                *bias = (LS) (highest); 
            } else {
                *bias = (LS) (lowest);
            }
        } else {
            *bias = tmp;
        }
        
    }
    

    
    result->cycles_compute += timer_stop(&cycles);

    barrier_wait(&my_barrier);

    

    // Write bias term
    if (tasklet_id == 15) {
        timer_start(&cycles);

        mram_write((void *) bias,(__mram_ptr void*) (mram_base_addr_bias), 8);
        result->cycles_wram_to_mram += timer_stop(&cycles);
        result->bytes_wram_to_mram += (uint64_t) (8);
    }

    if (tasklet_id == 0) {
        DIA.current_batch_id = DIA.current_batch_id + 1;
    }
    

    // Barrier
    barrier_wait(&my_barrier);

    return 0;
}
