#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>


#include "../common_support/common.h"

#include "../minimal/_tpl_only_ldexpf_exp/dpu/lut_exp.h"

__host dpu_arguments_t DIA; 
__mram_noinit S DPU_RESULTS[N_FEATURES]; 

__dma_aligned LS dot_product_frac_tmp[NR_TASKLETS]; 
__dma_aligned uint32_t mram_base_addr_Y;
__dma_aligned U cache_Y[2]; 
__dma_aligned LS diff_tmp;
__dma_aligned uint32_t index_Y;



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
    }
    barrier_wait(&my_barrier); // Barrier

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
    uint32_t current_batch_id = DIA.current_batch_id;

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
    uint32_t batch_size_frac_byte_Y = b_size_frac << MUL_U;
    uint32_t batch_size_frac_pad_byte_Y = batch_size_frac_Y_pad << MUL_U;

    uint32_t communicate_Y_pad_byte = communicate_Y_pad << MUL_U;

    uint32_t learning_rate = DIA.learning_rate;
    uint32_t shift_plus_learning_rate = learning_rate + SHIFT_AMOUNT;


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

    
    uint32_t mram_base_addr_X = (uint32_t) (DPU_MRAM_HEAP_POINTER + current_batch_id*mram_batch_offset+ start_index_byte); 

    if (tasklet_id == 0) {
        mram_base_addr_Y = (uint32_t) (DPU_MRAM_HEAP_POINTER + mram_offset + current_batch_id*batch_size_frac_byte_Y); //the labels that are relevant for this tasklet
    }

    uint32_t mram_base_addr_W = (uint32_t) (DPU_MRAM_HEAP_POINTER + mram_offset + communicate_Y_pad_byte + start_index_byte);
    uint32_t mram_base_addr_dpu_results = (uint32_t) (*tmp_DPU_RESULTS) + start_index_byte;
 
    

    uint32_t mram_temp_addr_X = mram_base_addr_X;
    uint32_t mram_temp_addr_Y = mram_base_addr_Y;
    uint32_t mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
    uint32_t mram_temp_addr_W = mram_base_addr_W;

    for (uint32_t i = 0; i < samples_loop_transfer; i++) {
        for (uint32_t k = 0; k < samples_per_transfer; ++k) {
            cache_dpu_results[k] = (S) (0);
        }
        mram_write((void *) cache_dpu_results,(__mram_ptr void*) (mram_temp_addr_dpu_results), transfer);
        mram_temp_addr_dpu_results += transfer;
    }
    mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
    

    
    if (tasklet_id == 15) {
        *bias_gradient = (LS) (0);
    }
    barrier_wait(&my_barrier);

    for (uint32_t batch_size_fraction_id = 0; batch_size_fraction_id < b_size_frac; ++batch_size_fraction_id) {

        mram_temp_addr_X = mram_base_addr_X + batch_size_fraction_id * n_size_byte; 

        if (tasklet_id == 0) {
            if (batch_size_fraction_id % 2 == 0) {
                index_Y = 0;
                mram_read((__mram_ptr void const*) (mram_temp_addr_Y), (void*) cache_Y, 8); 
                mram_temp_addr_Y += 8;
            }
            
        }
        
        mram_read((__mram_ptr void const*) (mram_temp_addr_X), cache_X, n_size_features_frac_pad_byte); 
        dot_product_frac_tmp[tasklet_id] = (LS) (0);
        for (uint32_t i = 0; i < samples_loop_transfer; i++) {
            mram_read((__mram_ptr void const*) (mram_temp_addr_W), (void *) cache_W, transfer); 
            dot_product_frac_tmp[tasklet_id] += dot_product(cache_X, cache_W, samples_per_transfer, i*samples_per_transfer);
            mram_temp_addr_W += transfer;
        }
        mram_temp_addr_W = mram_base_addr_W; 

        if (tasklet_id == 15) {
            dot_product_frac_tmp[tasklet_id] += *bias;
        }

        barrier_wait(&my_barrier);
        if (tasklet_id == 0) { 
            diff_tmp = (S) (0);

            LS dot_product_tmp = (LS) (0);
            for (uint32_t k = 0; k < NR_TASKLETS; ++k) {
                dot_product_tmp += dot_product_frac_tmp[k];
            }
            
            LS sigmoid = (LS) (0);
            float exp_tmp = expf(-((float) dot_product_tmp / CAST));
            
            sigmoid = (LS)  (CAST / (1.0 + exp_tmp));
            LS label = cache_Y[index_Y] == 1 ? ((LS) (CAST)) : ((LS) (0));

            diff_tmp = (LS) (sigmoid - label);
            ++index_Y;
        }
        barrier_wait(&my_barrier);
        
        uint32_t gradient_index_tmp = 0;
        LS feature_tmp = (LS) (0); 
        S check_tmp = (S) (0);
        for (uint32_t i = 0; i < samples_loop_transfer; ++i) {
            mram_read((__mram_ptr void*) (mram_temp_addr_dpu_results), (void *) cache_dpu_results, transfer);

            for (uint32_t k = 0; k < samples_per_transfer; ++k) {
                feature_tmp = (LS) cache_X[gradient_index_tmp];
                check_tmp = (S) ( (feature_tmp * diff_tmp) >> (shift_plus_learning_rate) );
                if (check_tmp != -1) {
                    cache_dpu_results[k] += check_tmp;
                }
                
                ++gradient_index_tmp;
            }
            mram_write((void *) cache_dpu_results,(__mram_ptr void*) (mram_temp_addr_dpu_results), transfer);
            mram_temp_addr_dpu_results += transfer;
        }
        mram_temp_addr_W = mram_base_addr_W;
        mram_temp_addr_dpu_results = mram_base_addr_dpu_results;
        

         
        // update bias_gradient
        if (tasklet_id == 15) {
            LS check_tmp_bias = (LS) (diff_tmp >> learning_rate);
            if (check_tmp_bias != -1) {
                *bias_gradient += check_tmp_bias;
            }
        }
    } // iteration over all elements of b_size_frac done

            

    
    
    // Write bias term
    if (tasklet_id == 15) {
        mram_write((void *) bias_gradient,(__mram_ptr void*) (mram_base_addr_bias), 8);
    }

    if (tasklet_id == 0) {
        DIA.current_batch_id = DIA.current_batch_id + 1;
    }
    

    // Barrier
    barrier_wait(&my_barrier);

    return 0;
}
