#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#include <pthread.h>

#include <math.h>
#include <time.h>


#include "../common_support/common.h"


// include host_utils
#include "../host_utils/read_data_utils.h"

#include "../minimal/_tpl_only_ldexpf_exp/host/lut_exp_host.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

static U *X;
static U *Y;

static U *X_test;
static U *Y_test;

void* model_average_function(void* thread_info_list){
    model_average_parallel_t* thread_info_list_current = (model_average_parallel_t*) thread_info_list;
    LS scale = thread_info_list_current->scale;
    uint64_t thread_id = thread_info_list_current->thread_id;
    LS overflow_test = thread_info_list_current->overflow_test;
    LS underflow_test = thread_info_list_current->underflow_test;

    if (thread_id < 16) { 
        LS* W_ADMM_global_buffer = thread_info_list_current->W_ADMM_global_buffer;
        LS* U_ADMM_global_buffer = thread_info_list_current->U_ADMM_global_buffer;

        S* W_ADMM_global = thread_info_list_current->W_ADMM_global;
        S* U_ADMM_global = thread_info_list_current->U_ADMM_global;
        S* W_ADMM_local = thread_info_list_current->W_ADMM_local;
        S* Z_ADMM_global = thread_info_list_current->Z_ADMM_global;
        S* U_ADMM_local = thread_info_list_current->U_ADMM_local;
        S* U_Z_ADMM_local = thread_info_list_current->U_Z_ADMM_local;


        uint64_t number_of_features_per_thread = thread_info_list_current->number_of_features_per_thread;
        uint64_t n_size = thread_info_list_current->n_size;

        // Model average W_ADMM
        for (uint64_t m = 0; m < NR_DPUS; ++m) {
            for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
                W_ADMM_global_buffer[x] += (LS) W_ADMM_local[m*n_size + x];
            }
        }
        
        for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
            LS tmp = W_ADMM_global_buffer[x] / NR_DPUS;
            if (tmp > overflow_test || tmp < underflow_test) {
                if (tmp > overflow_test) {
                    W_ADMM_global[x] = (S) (overflow_test - 1);
                } else {
                    W_ADMM_global[x] = (S) (underflow_test + 1);
                }
            } else {
                W_ADMM_global[x] = (S) (tmp);
            }
        }
        // Model average U_ADMM
        for (uint64_t m = 0; m < NR_DPUS; ++m) {
            for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
                U_ADMM_global_buffer[x] += (LS) U_ADMM_local[m*n_size + x];
            }
        }
        
        for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
            LS tmp = U_ADMM_global_buffer[x] / NR_DPUS;
            if (tmp > overflow_test || tmp < underflow_test) {
                if (tmp > overflow_test) {
                    U_ADMM_global[x] = (S) (overflow_test - 1);
                } else {
                    U_ADMM_global[x] = (S) (underflow_test + 1);
                }
            } else {
                U_ADMM_global[x] = (S) (tmp);
            }
        }
        // Compute Z_ADMM_global: soft_threshholding_operation
        for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
            LS tmp = (LS) (((LS) (W_ADMM_global[x])) + ((LS) (U_ADMM_global[x])));
            if (tmp > overflow_test || tmp < underflow_test) {
                if (tmp > overflow_test) {
                    tmp = (LS) (overflow_test - 1);
                } else {
                    tmp = (LS) (underflow_test + 1);
                }
            }
            
            LS check_Z_ADMM_global = (LS) ((scale * tmp) >> SHIFT_AMOUNT);
            
            if (check_Z_ADMM_global > overflow_test || check_Z_ADMM_global < underflow_test) {
                if (check_Z_ADMM_global > overflow_test) {
                    Z_ADMM_global[x] = (S) (overflow_test - 1);
                } else {
                    Z_ADMM_global[x] = (S) (underflow_test + 1);
                }
            } else {
                Z_ADMM_global[x] = (S) (check_Z_ADMM_global);
            }

        }
        // Compute U_ADMM_local
        for (uint64_t m = 0; m < NR_DPUS; ++m) {
            uint32_t index = m * n_size;
            for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
                LS tmp = (LS) (((LS) U_ADMM_local[index + x]) + (((LS) W_ADMM_local[index + x]) - ((LS) Z_ADMM_global[x])));
                if (tmp > overflow_test || tmp < underflow_test) {
                    if (tmp > overflow_test) {
                        U_ADMM_local[index + x] = (S) (overflow_test - 1);
                    } else {
                        U_ADMM_local[index + x] = (S) (underflow_test + 1);
                    }
                } else {
                    U_ADMM_local[index + x] = (S) (tmp);
                }
            }
        }
        // Compute U_Z_ADMM_local
        for (uint64_t m = 0; m < NR_DPUS; ++m) {
            uint32_t index = m * n_size;
            for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
                LS tmp = (LS) (((LS) U_ADMM_local[index + x]) - ((LS) Z_ADMM_global[x]));
                if (tmp > overflow_test || tmp < underflow_test) {
                    if (tmp > overflow_test) {
                        U_Z_ADMM_local[index + x] = (S) (overflow_test - 1);
                    } else {
                        U_Z_ADMM_local[index + x] = (S) (underflow_test + 1);
                    }
                } else {
                    U_Z_ADMM_local[index + x] = (S) (tmp);
                }
            }
        }

        
    } else {
        LS* bias_W_ADMM_global_buffer = thread_info_list_current->bias_W_ADMM_global_buffer;
        LS* bias_U_ADMM_global_buffer = thread_info_list_current->bias_U_ADMM_global_buffer;

        LS* bias_W_ADMM_global = thread_info_list_current->bias_W_ADMM_global;
        LS* bias_U_ADMM_global = thread_info_list_current->bias_U_ADMM_global;
        LS* bias_W_ADMM_local = thread_info_list_current->bias_W_ADMM_local;
        LS* bias_Z_ADMM_global = thread_info_list_current->bias_Z_ADMM_global;
        LS* bias_U_ADMM_local = thread_info_list_current->bias_U_ADMM_local;
        LS* bias_U_Z_ADMM_local = thread_info_list_current->bias_U_Z_ADMM_local;


        // Model average bias_W_ADMM
        for (uint64_t m = 0; m < NR_DPUS; ++m) {
            *bias_W_ADMM_global_buffer += bias_W_ADMM_local[m];
        }
        LS tmp = *bias_W_ADMM_global_buffer / NR_DPUS;
        if (tmp > overflow_test || tmp < underflow_test) {
            if (tmp > overflow_test) {
                *bias_W_ADMM_global = (LS) (overflow_test - 1);
            } else {
                *bias_W_ADMM_global = (LS) (underflow_test + 1);
            }
        } else {
            *bias_W_ADMM_global = tmp;
        }
        // Model average bias_U_ADMM
        for (uint64_t m = 0; m < NR_DPUS; ++m) {
            *bias_U_ADMM_global_buffer += bias_U_ADMM_local[m];
        }
        tmp = *bias_U_ADMM_global_buffer / NR_DPUS;
        if (tmp > overflow_test || tmp < underflow_test) {
            if (tmp > overflow_test) {
                *bias_U_ADMM_global = (LS) (overflow_test - 1);
            } else {
                *bias_U_ADMM_global = (LS) (underflow_test + 1);
            }
        } else {
            *bias_U_ADMM_global = tmp;
        }
        // Compute bias_Z_ADMM: soft_threshholding_operation
        tmp = (LS) ((*bias_W_ADMM_global) + (*bias_U_ADMM_global));
        if (tmp > overflow_test || tmp < underflow_test) {
            if (tmp > overflow_test) {
                tmp = (LS) (overflow_test - 1);
            } else {
                tmp = (LS) (underflow_test + 1);
            }
        }
        LS check_Z_ADMM_global = (LS) ((scale * tmp) >> SHIFT_AMOUNT);
        if (check_Z_ADMM_global > overflow_test || check_Z_ADMM_global < underflow_test) {
            if (check_Z_ADMM_global > overflow_test) {
                *bias_Z_ADMM_global = (LS) (overflow_test - 1);
            } else {
                *bias_Z_ADMM_global = (LS) (underflow_test + 1);
            }
        } else {
            *bias_Z_ADMM_global = check_Z_ADMM_global;
        }
        // Compute bias_U_ADMM_local
        for (uint64_t m = 0; m < NR_DPUS; ++m) {
            tmp = (LS) (bias_U_ADMM_local[m] + (bias_W_ADMM_local[m] - (*bias_Z_ADMM_global)));
            if (tmp > overflow_test || tmp < underflow_test) {
                if (tmp > overflow_test) {
                    bias_U_ADMM_local[m] = (LS) (overflow_test - 1);
                } else {
                    bias_U_ADMM_local[m] = (LS) (underflow_test + 1);
                }
            } else {
                bias_U_ADMM_local[m] = tmp;
            }
        }
        // Compute bias_U_Z_ADMM_local
        for (uint64_t m = 0; m < NR_DPUS; ++m) {
            tmp = (LS) (bias_U_ADMM_local[m] - (*bias_Z_ADMM_global));
            if (tmp > overflow_test || tmp < underflow_test) {
                if (tmp > overflow_test) {
                    bias_U_Z_ADMM_local[m] = (LS) (overflow_test - 1);
                } else {
                    bias_U_Z_ADMM_local[m] = (LS) (underflow_test + 1);
                }
            } else {
                bias_U_Z_ADMM_local[m] = tmp;
            }
        }

    


    }
    

    pthread_exit(NULL);
}

void* compute_error_rate_parallel(void* thread_info_list){
    compute_error_rate_worker_t* thread_info_list_current = (compute_error_rate_worker_t*) thread_info_list;
    float* error_rate = thread_info_list_current->error_rate;
    uint64_t* reduction = thread_info_list_current->reduction;
    uint64_t* sum_of_Y = thread_info_list_current->sum_of_Y;
    U* X = thread_info_list_current->X;
    U* Y = thread_info_list_current->Y;
    float* W_host_float = thread_info_list_current->W_host_float;
    float* bias_global_model_buffer = thread_info_list_current->bias_global_model_buffer;
    uint64_t m_size = thread_info_list_current->m_size;
    uint64_t n_size = thread_info_list_current->n_size;

    for (uint64_t m = 0; m < m_size; ++m) {
        float dot_product = 0.0;
        for (uint64_t n = 0; n < n_size; ++n) {
            dot_product += ((float) X[m*n_size + n] / CAST)*W_host_float[n]; 
        }
        dot_product += *bias_global_model_buffer;
        
        S predict_temp = dot_product < 0 ? ((S) 0) : ((S) 1); // 0 represents -1 for SVM
        S label = Y[m] == 1 ? ((S) 1):((S) 0);
        if(predict_temp != label){
            (*reduction)++; 
        }
        if (Y[m] == 1) {
            *sum_of_Y += 1; 
        }
    }
    *error_rate = ((float) ((float) *reduction / m_size)*100);

    pthread_exit(NULL);
}

void* compute_hinge_loss_parallel(void* thread_info_list) {
    compute_loss_worker_t* thread_info_list_current = (compute_loss_worker_t*) thread_info_list;
    double* hinge_loss = thread_info_list_current->hinge_loss;
    U* X = thread_info_list_current->X;
    U* Y = thread_info_list_current->Y;
    float* W_host_float = thread_info_list_current->W_host_float;
    float* bias_global_model_buffer = thread_info_list_current->bias_global_model_buffer;
    uint64_t m_size = thread_info_list_current->m_size;
    uint64_t n_size = thread_info_list_current->n_size;

    for (uint64_t m = 0; m < m_size; ++m) {
        double dot_product = 0.0;
        for (uint64_t n = 0; n < n_size; ++n) {
            dot_product += (double) (((float) X[m*n_size + n] / CAST)*W_host_float[n]);
        }
        double bias_global_model_buffer_double_tmp = (double) (*bias_global_model_buffer);
        dot_product += bias_global_model_buffer_double_tmp;
        double hinge_loss_tmp = Y[m] == 1 ? ((double) (1.0)) : ((double) (-1.0));
        hinge_loss_tmp = (double) (((double) (0.5)) - hinge_loss_tmp * dot_product); // CONSTRAINT (which corresponds to 0.5) instead of 1.0
        if (hinge_loss_tmp >= 0) {
            *hinge_loss += hinge_loss_tmp;
        }
        
    }


    pthread_exit(NULL);
}

// Main of the Host Application
int main(int argc, char **argv) {
    

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;

    struct timespec start_main, end_main;
    double elapsed_time_main; 

    clock_gettime(CLOCK_MONOTONIC, &start_main);
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set)); 
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    clock_gettime(CLOCK_MONOTONIC, &end_main);
    printf("allocated NR_DPUS %u\n", NR_DPUS);
    printf("loaded binary to DPUs\n");
    printf("Loaded binary to dpus and allocated dpus\n");
    elapsed_time_main = (double) ((end_main.tv_sec - start_main.tv_sec) + (end_main.tv_nsec - start_main.tv_nsec) / 1e9);
    printf("elapsed time is %.9f\n", elapsed_time_main);

    uint64_t strong_scaling = 0;

    uint32_t array_b_size_frac[] = {8, 16, 32, 64}; 
    uint32_t array_b_size_frac_length = sizeof(array_b_size_frac) / sizeof(array_b_size_frac[0]);

    if (AE == 1) { // reproducing the results in our paper
        array_b_size_frac[0] = 8;
        array_b_size_frac_length = 1;
    }

    uint32_t array_learning_rate[] = {6};
    uint32_t array_learning_rate_length = sizeof(array_learning_rate) / sizeof(array_learning_rate[0]);
    
    uint32_t array_reg_term[] = {10}; 
    uint32_t array_reg_term_length = sizeof(array_reg_term) / sizeof(array_reg_term[0]);

    uint32_t array_reg_term_alpha[] = {12}; // reg_term_alpha in ADMM jargon corresponds to alpha
    uint32_t array_reg_term_alpha_length = sizeof(array_reg_term_alpha) / sizeof(array_reg_term_alpha[0]);

    uint32_t array_epochs[] = {1};
    uint32_t array_epochs_length = sizeof(array_epochs)/sizeof(array_epochs[0]);
    uint32_t max_epochs_length = array_epochs[array_epochs_length-1];
    uint32_t array_task_epochs[] = {1}; 
    uint32_t array_task_epochs_length = sizeof(array_task_epochs) / sizeof(array_task_epochs[0]);

    uint64_t m_size = (uint64_t) (0);
    uint64_t number_of_test_samples = (uint64_t) (0);
    if (NR_DPUS == 256 || strong_scaling == 1) {
        m_size = (uint64_t) (851968);
        number_of_test_samples = (uint64_t) (212992);
    }
    else if (NR_DPUS == 512) {
        m_size = (uint64_t) (1703936);
        number_of_test_samples = (uint64_t) (425984);
    }
    else if (NR_DPUS == 1024) {
        m_size = (uint64_t) (3407872);
        number_of_test_samples = (uint64_t) (851968);
    }
    else if (NR_DPUS == 2048) {
        m_size = (uint64_t) (6815744);
        number_of_test_samples = (uint64_t) (1703936);
    }

    uint32_t n_size = N_FEATURES;

    uint32_t n_size_pad = ((n_size*sizeof(U)) % 8) == 0 ? n_size : roundup(n_size, (8/sizeof(U)));
    uint32_t n_size_pad_W_fp = ((n_size*sizeof(S)) % 8) == 0 ? n_size : roundup(n_size, (8/sizeof(S)));

    uint32_t n_size_features_frac = n_size/NR_TASKLETS;
    uint32_t n_size_features_frac_pad = ((n_size_features_frac*sizeof(U)) % 8) == 0 ? n_size_features_frac : roundup(n_size_features_frac, (8/sizeof(U)));

    // Input/output allocation
    X = calloc((uint64_t) (m_size * ((uint64_t) n_size_pad)), sizeof(U)); 
    Y = calloc(m_size, sizeof(U)); 

    X_test = malloc(number_of_test_samples * ((uint64_t) n_size_pad) * ((uint64_t) sizeof(U)));
    Y_test = malloc(number_of_test_samples * sizeof(U));
    
    // init trainging dataset and weight for host 
    U *bufferX = X;
    U *bufferY = Y;

    U *bufferX_test = X_test;
    U *bufferY_test = Y_test;

    printf("Before reading test input\n");
    printf("\nBefore reading train input\n");
    read_input_yfcc100m(bufferX, bufferY, m_size, strong_scaling);
    
    
    printf("Before starting the experiments\n");
    char* filename_experiment_dpu = calloc(1000, sizeof(char));
    float* W_ADMM_checkpoint_for_compute_error = calloc(max_epochs_length*n_size, sizeof(float));
    float* Z_ADMM_checkpoint_for_compute_error = calloc(max_epochs_length*n_size, sizeof(float));
    float* bias_W_ADMM_checkpoint_for_compute_error = calloc(max_epochs_length*1, sizeof(float));
    float* bias_Z_ADMM_checkpoint_for_compute_error = calloc(max_epochs_length*1, sizeof(float));
    float* compute_error_rate_worker_list = calloc(64, sizeof(float));
    uint64_t* reduction_worker_list = calloc(64, sizeof(uint64_t));
    uint64_t* sum_of_Y_worker_list = calloc(64, sizeof(uint64_t));
    double* compute_loss_worker_list = calloc(64, sizeof(double));
    float* W_ADMM_global_float_error = calloc(n_size, sizeof(float));
    float* Z_ADMM_global_float_error = calloc(n_size, sizeof(float));
    float* bias_W_ADMM_dpu_float_error = calloc(1, sizeof(float));
    float* bias_Z_ADMM_dpu_float_error = calloc(1, sizeof(float));
    LS* W_ADMM_global_buffer = (LS*) calloc(n_size, sizeof(LS));
    LS* U_ADMM_global_buffer = (LS*) calloc(n_size, sizeof(LS));
    LS* bias_W_ADMM_global_buffer = (LS*) calloc(1, sizeof(LS));
    LS* bias_U_ADMM_global_buffer = (LS*) calloc(1, sizeof(LS));

    compute_error_rate_worker_t* thread_info_compute_error_rate_list = (compute_error_rate_worker_t*) calloc(64, sizeof(compute_error_rate_worker_t)); // max number of threads is 64
    compute_loss_worker_t* thread_info_compute_loss_list = (compute_loss_worker_t*) calloc(64, sizeof(compute_loss_worker_t));

    // set compute_error_rate_worker_list to zero again after usage
    // set compute_loss_worker_list to zero again after usage
    uint64_t start_index_worker_train = ((uint64_t) m_size) / ((uint64_t) 64);
    uint64_t start_index_worker_test = ((uint64_t) number_of_test_samples) / ((uint64_t) 64);
    for (uint64_t s = 0; s < 64; ++s) {
        thread_info_compute_error_rate_list[s].error_rate = compute_error_rate_worker_list + s;
        thread_info_compute_error_rate_list[s].reduction = reduction_worker_list + s;
        thread_info_compute_error_rate_list[s].sum_of_Y = sum_of_Y_worker_list + s;
        thread_info_compute_error_rate_list[s].m_size = start_index_worker_train;
        thread_info_compute_error_rate_list[s].n_size = (uint64_t) n_size;

        thread_info_compute_loss_list[s].hinge_loss = compute_loss_worker_list + s;
        thread_info_compute_loss_list[s].m_size = start_index_worker_train;
        thread_info_compute_loss_list[s].n_size = (uint64_t) n_size;
    }

    
    float* error_rate = calloc(1, sizeof(float));
    uint64_t* reduction = calloc(1, sizeof(uint64_t));
    uint64_t* sum_of_Y = calloc(1, sizeof(uint64_t));
    double* hinge_loss = calloc(1, sizeof(double));

    for (uint32_t ie = 0; ie < array_b_size_frac_length; ++ie) {
        uint32_t b_size_frac = array_b_size_frac[ie];
        printf("Initialized b_size_frac\n");
        for (uint32_t je = 0; je < array_learning_rate_length; ++je) {
            uint32_t learning_rate = array_learning_rate[je];
            printf("Initialized learning rate\n");
            for (uint32_t ke = 0; ke < array_epochs_length; ++ke) {
                uint32_t epochs = array_epochs[ke];
                printf("Initialized epochs\n");

                for(uint32_t ee = 0; ee < array_reg_term_length; ++ee) { 
                    uint32_t reg_term = array_reg_term[ee];
                    printf("Initialized reg_term\n");
                    for (uint32_t se = 0; se < array_reg_term_alpha_length; ++se) { 
                        uint32_t reg_term_alpha = array_reg_term_alpha[se];
                        printf("Initialized reg_term_alpha\n");
                        for (uint32_t le = 0; le < array_task_epochs_length; ++le) {
                            uint32_t task_epochs = array_task_epochs[le];
                            printf("Initialized task_epochs\n");
                            uint32_t nr_batches = (uint32_t) m_size/(b_size_frac*NR_DPUS);
                            
                            memset(filename_experiment_dpu, 0, 1000*sizeof(char));
                            memset(W_ADMM_checkpoint_for_compute_error, 0, max_epochs_length*n_size*sizeof(float));
                            memset(Z_ADMM_checkpoint_for_compute_error, 0, max_epochs_length*n_size*sizeof(float));
                            memset(bias_W_ADMM_checkpoint_for_compute_error, 0, max_epochs_length*1*sizeof(float));
                            memset(bias_Z_ADMM_checkpoint_for_compute_error, 0, max_epochs_length*1*sizeof(float));
                            memset(W_ADMM_global_float_error, 0, n_size*sizeof(float));
                            memset(Z_ADMM_global_float_error, 0, n_size*sizeof(float));
                            memset(bias_W_ADMM_dpu_float_error, 0, sizeof(float));
                            memset(bias_Z_ADMM_dpu_float_error, 0, sizeof(float));
                            memset(W_ADMM_global_buffer, 0, n_size*sizeof(LS));
                            memset(U_ADMM_global_buffer, 0, n_size*sizeof(LS));
                            memset(bias_W_ADMM_global_buffer, 0, sizeof(LS));
                            memset(bias_U_ADMM_global_buffer, 0, sizeof(LS));
                            memset(thread_info_compute_error_rate_list, 0, 64*sizeof(compute_error_rate_worker_t));
                            memset(thread_info_compute_loss_list, 0, 64*sizeof(compute_loss_worker_t));
                            memset(error_rate, 0, sizeof(float));
                            memset(reduction, 0, sizeof(uint64_t));
                            memset(sum_of_Y, 0, sizeof(uint64_t));
                            memset(hinge_loss, 0, sizeof(double));


                            printf("About to create fdpu file\n");
                            sprintf(filename_experiment_dpu, "%s/benchmark_UPMEM/YFCC100M-HNfc6/admm_SVM_uint32/svm_reg_DPU_arch__yfcc100m__uint32__ADMM__NR_DPUS_%u__NR_TASKLETS_%u__m_size_%lu__m_test_size_%lu__n_size_%u__b_size_frac_%u__nr_batches_%u__learning_rate_%u__reg_term_%u__alpha_%u__epochs_%u__task_epochs_%u.txt", DEST_DIR, NR_DPUS, NR_TASKLETS, m_size, number_of_test_samples, n_size, b_size_frac, nr_batches, learning_rate, reg_term, reg_term_alpha, epochs, task_epochs);

                            FILE *fp_dpu = fopen(filename_experiment_dpu, "w");
                            if (fp_dpu == NULL) {
                                printf("Could not open file %s\n", filename_experiment_dpu);
                            }
                            fprintf(fp_dpu, "Experiment successfully started.\n");
                            time_t timestamp;
                            struct tm *tmp;
                            timestamp = time(NULL);
                            tmp = localtime(&timestamp);
                            fprintf(fp_dpu, "Date: %02d.%02d.%d, %02d:%02d\n", tmp->tm_mday, tmp->tm_mon + 1, tmp->tm_year + 1900, tmp->tm_hour, tmp->tm_min);



                            printf("fdpu file created\n");
                            fprintf(fp_dpu, "elapsed time for allocation of DPUs, loading binary to DPUs, and dpu_get_nr_dpus. Elapsed time is %.9f s\n", elapsed_time_main);

                            // Measurement
                            struct timespec start, end;
                            double elapsed_time;

                            // Initialize help data
                            clock_gettime(CLOCK_MONOTONIC, &start);
                            dpu_info_t *dpu_info = (dpu_info_t *) malloc(nr_of_dpus * sizeof(dpu_info_t)); 
                            dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
                            uint32_t max_rows_per_dpu = 0;
                            
                            uint32_t batch_size_frac_Y_pad = (b_size_frac*sizeof(U) % 8) == 0 ? b_size_frac : roundup(b_size_frac, (8/sizeof(U)));

                            uint32_t size_of_batch = b_size_frac * nr_of_dpus;

                            uint32_t samples_loop_transfer = (U) (2);
                            uint32_t transfer = (U) (512);
                            uint32_t samples_per_transfer = (U) (128);


                            unsigned int i = 0;
                            DPU_FOREACH(dpu_set, dpu, i) {
                                uint32_t rows_per_dpu;
                                uint32_t prev_rows_dpu = 0;
                                uint32_t chunks = m_size / nr_of_dpus;
                                rows_per_dpu = chunks;
                                uint32_t rest_rows = m_size % nr_of_dpus;
                                
                                if (rest_rows > 0) {
                                    if (i > rest_rows) {
                                        prev_rows_dpu = rest_rows + i * chunks;
                                    } else if (i <= rest_rows) {
                                        prev_rows_dpu = i * (chunks + b_size_frac);
                                    } 
                                } else {
                                    prev_rows_dpu = i * chunks;
                                }
                                

                                // Keep max rows for parallel transfers
                                uint32_t rows_per_dpu_pad = ((rows_per_dpu*sizeof(U)) % 8) == 0 ? rows_per_dpu : roundup(rows_per_dpu, (8/sizeof(U))); 
                                if (rows_per_dpu_pad > max_rows_per_dpu)
                                    max_rows_per_dpu = rows_per_dpu_pad;

                                dpu_info[i].rows_per_dpu = rows_per_dpu;
                                dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
                                dpu_info[i].prev_rows_dpu = prev_rows_dpu;


                                // Copy input arguments to DPU
                                input_args[i].n_size = n_size;
                                input_args[i].n_size_pad = n_size_pad;
                                input_args[i].n_size_pad_W_fp = n_size_pad_W_fp;
                                input_args[i].n_size_features_frac = n_size_features_frac;
                                input_args[i].n_size_features_frac_pad = n_size_features_frac_pad;

                                input_args[i].batch_size_frac_Y_pad = batch_size_frac_Y_pad;
                                input_args[i].current_batch_id = 0;

                                input_args[i].nr_batches = nr_batches;
                                input_args[i].size_of_batch = size_of_batch;
                                input_args[i].learning_rate = learning_rate; 
                                input_args[i].reg_term = reg_term;
                                input_args[i].learning_rate_plus_reg_term = (U) (learning_rate + reg_term);
                                input_args[i].task_epochs = task_epochs;
                                input_args[i].b_size_frac = b_size_frac;
                                input_args[i].b_size_frac_log = (U) abs((int32_t)log2(b_size_frac));

                                input_args[i].samples_loop_transfer = samples_loop_transfer;
                                input_args[i].transfer = transfer;
                                input_args[i].samples_per_transfer = samples_per_transfer;

                            }
                            printf("Current learning rate: %u\n", input_args[127].learning_rate);
                            printf("Current b_size_frac: %u\n", input_args[127].b_size_frac_log);

                            // init Weight for DPU 
                            S* W_ADMM_global = calloc(n_size_pad_W_fp, sizeof(S)); 
                            S* U_ADMM_global = calloc(n_size_pad_W_fp, sizeof(S)); 
                            S* W_ADMM_local = (S*) calloc(NR_DPUS*n_size, sizeof(S));
                            S* Z_ADMM_global = (S*) calloc(n_size, sizeof(S));
                            S* U_ADMM_local = (S*) calloc(NR_DPUS*n_size, sizeof(S));
                            S* U_Z_ADMM_local = (S*) calloc(NR_DPUS*n_size, sizeof(S));

                            LS* bias_W_ADMM_global = (LS*) calloc(1, sizeof(LS));
                            LS* bias_U_ADMM_global = (LS*) calloc(1, sizeof(LS));
                            LS* bias_W_ADMM_local = (LS*) calloc(NR_DPUS, sizeof(LS));
                            LS* bias_Z_ADMM_global = (LS*) calloc(1, sizeof(LS));
                            LS* bias_U_ADMM_local = (LS*) calloc(NR_DPUS, sizeof(LS));
                            LS* bias_U_Z_ADMM_local = (LS*) calloc(NR_DPUS, sizeof(LS));

                            LS overflow_test = (LS) CAST;
                            LS underflow_test = -overflow_test;
                            

                            LS scale = (LS) (0);
                            uint32_t K_shift = (uint32_t) (0); // see paper, K corresponds to number of models
                            if (NR_DPUS == 256) {
                                K_shift = (uint32_t) (8);
                            } else if (NR_DPUS == 512) {
                                K_shift = (uint32_t) (9);
                            } else if (NR_DPUS == 1024) {
                                K_shift = (uint32_t) (10);
                            } else if (NR_DPUS == 2048) {
                                K_shift = (uint32_t) (11);
                            }
                            LS scale_nominator = (LS) (((LS) (CAST)) >> reg_term);
                            LS scale_denominator = (LS) (((LS) (CAST)) << reg_term_alpha);
                            if (reg_term > K_shift) {
                                scale_denominator += (LS) (((LS) (CAST)) >> (reg_term-K_shift));
                            } else {
                                scale_denominator += (LS) (((LS) (CAST)) << (K_shift-reg_term));
                            }
                            scale_nominator = (LS) (scale_nominator << SHIFT_AMOUNT);
                            scale = (LS) (scale_nominator/scale_denominator);
                            // model average with 16 threads
                            model_average_parallel_t* thread_info_model_average_parallel_list = (model_average_parallel_t*) calloc(17, sizeof(model_average_parallel_t));
                            uint64_t start_index_thread_model_average = (uint64_t) (256);
                            for (uint64_t s = 0; s < 17; ++s) {
                                thread_info_model_average_parallel_list[s].scale = scale;
                                thread_info_model_average_parallel_list[s].thread_id = s;
                                thread_info_model_average_parallel_list[s].overflow_test = overflow_test;
                                thread_info_model_average_parallel_list[s].underflow_test = underflow_test;

                                if (s < 16) {
                                    thread_info_model_average_parallel_list[s].W_ADMM_global_buffer = W_ADMM_global_buffer + s * start_index_thread_model_average;
                                    thread_info_model_average_parallel_list[s].U_ADMM_global_buffer = U_ADMM_global_buffer + s * start_index_thread_model_average;

                                    thread_info_model_average_parallel_list[s].W_ADMM_global = W_ADMM_global + s * start_index_thread_model_average;
                                    thread_info_model_average_parallel_list[s].U_ADMM_global = U_ADMM_global + s * start_index_thread_model_average;
                                    thread_info_model_average_parallel_list[s].W_ADMM_local = W_ADMM_local + s * start_index_thread_model_average;
                                    thread_info_model_average_parallel_list[s].Z_ADMM_global = Z_ADMM_global + s * start_index_thread_model_average;
                                    thread_info_model_average_parallel_list[s].U_ADMM_local = U_ADMM_local + s * start_index_thread_model_average;
                                    thread_info_model_average_parallel_list[s].U_Z_ADMM_local = U_Z_ADMM_local + s * start_index_thread_model_average;

                                    thread_info_model_average_parallel_list[s].number_of_features_per_thread = (uint64_t) (256);
                                    thread_info_model_average_parallel_list[s].n_size = n_size_pad_W_fp;
                                } else {
                                    thread_info_model_average_parallel_list[s].bias_W_ADMM_global_buffer = bias_W_ADMM_global_buffer;
                                    thread_info_model_average_parallel_list[s].bias_U_ADMM_global_buffer = bias_U_ADMM_global_buffer;

                                    thread_info_model_average_parallel_list[s].bias_W_ADMM_global = bias_W_ADMM_global;
                                    thread_info_model_average_parallel_list[s].bias_U_ADMM_global = bias_U_ADMM_global;
                                    thread_info_model_average_parallel_list[s].bias_W_ADMM_local = bias_W_ADMM_local;
                                    thread_info_model_average_parallel_list[s].bias_Z_ADMM_global = bias_Z_ADMM_global;
                                    thread_info_model_average_parallel_list[s].bias_U_ADMM_local = bias_U_ADMM_local;
                                    thread_info_model_average_parallel_list[s].bias_U_Z_ADMM_local = bias_U_Z_ADMM_local;

                                }
                                
                                
                            }

                            clock_gettime(CLOCK_MONOTONIC, &end);
                            elapsed_time = (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                            fprintf(fp_dpu, "Preprocessing and initialization of variables. Elapsed time is %.9f s\n", elapsed_time);

                            dpu_result_t* results = (dpu_result_t*) calloc(NR_TASKLETS*NR_DPUS*epochs, sizeof(dpu_result_t));
                            dpu_result_double_t* results_double = (dpu_result_double_t*) calloc(NR_TASKLETS*NR_DPUS*epochs, sizeof(dpu_result_double_t));
                            uint64_t communications_init = (uint64_t) (4);
                            uint64_t communications_per_epoch = (uint64_t) (3);
                            uint64_t communications_CPU_DPU_per_epoch = (uint64_t) (2);
                            uint64_t communications_DPU_CPU_per_epoch = (uint64_t) (1);

                            double time_initialization = (double) (0);
                            double bytes_initialization = (double) (0);
                            double bandwidth_initialization = (double) (0);

                            

                            // Input arguments
                            clock_gettime(CLOCK_MONOTONIC, &start);
                            uint32_t communicate_Y_pad = ((max_rows_per_dpu * sizeof(U)) % 8) == 0 ? max_rows_per_dpu : roundup(max_rows_per_dpu, (8/sizeof(U)));


                            i = 0;
                            DPU_FOREACH(dpu_set, dpu, i) {
                                // Copy input arguments to DPU
                                input_args[i].max_rows = max_rows_per_dpu;
                                input_args[i].communicate_Y_pad = communicate_Y_pad;

                                DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
                            }
                            uint32_t dpu_arguments_size_pad = ((1 * sizeof(dpu_arguments_t))%8) == 0 ? 1 : roundup(1, (8/sizeof(dpu_arguments_t)));

                            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DIA", 0, dpu_arguments_size_pad*sizeof(dpu_arguments_t), \
                                DPU_XFER_DEFAULT)); 

                            // Copy X and y 
                            
                            i = 0;
                            DPU_FOREACH(dpu_set, dpu, i) {
                                DPU_ASSERT(dpu_prepare_xfer(dpu, bufferX + (((uint64_t) dpu_info[i].prev_rows_dpu) * ((uint64_t) n_size)))); 
                            }
                            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, \
                                max_rows_per_dpu * n_size_pad * sizeof(U), DPU_XFER_DEFAULT)); 
                            
                            i = 0;
                            DPU_FOREACH(dpu_set, dpu, i) {
                                DPU_ASSERT(dpu_prepare_xfer(dpu, bufferY + dpu_info[i].prev_rows_dpu)); 
                            }
                            DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                max_rows_per_dpu * n_size_pad * sizeof(U), communicate_Y_pad * sizeof(U), DPU_XFER_DEFAULT));

                            clock_gettime(CLOCK_MONOTONIC, &end);
                            elapsed_time = (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                            fprintf(fp_dpu, "Load input data to DPUs. Elapsed time is %.9f s\n", elapsed_time);

                            time_initialization += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                            bytes_initialization += (double) (((uint64_t) (NR_DPUS)) * ((uint64_t) ((dpu_arguments_size_pad*sizeof(dpu_arguments_t)) + (max_rows_per_dpu * n_size_pad * sizeof(U)) + (communicate_Y_pad * sizeof(U)))));
                            bytes_initialization = (double) (bytes_initialization / ((double) (1e9)));
                            bandwidth_initialization = (double) (bytes_initialization/time_initialization);
                            
                            fprintf(fp_dpu, "Benchmark:CPU_DPU_initialization_bandwidth = %.9f GB/s, CPU_DPU_initialization_elapsed_time = %.9f s, CPU_DPU_initialization_giga_bytes = %.9f GB\n\n", bandwidth_initialization,time_initialization,bytes_initialization);

                            double clock_speed_DPU = (double) (((double)(350)) * ((double) (1e6)));
                            uint64_t clock_speed_DPU_uint64 = (uint64_t) (((uint64_t)(350)) * ((uint64_t) (1e6)));
                            double epochs_double = (double) (epochs);
                            double epoch_time_CPU_DPU = (double) (0);
                            double epoch_bytes_CPU_DPU = (double) (0);
                            double epoch_bandwidth_CPU_DPU = (double) (0);
                            double epoch_time_DPU_CPU = (double) (0);
                            double epoch_bytes_DPU_CPU = (double) (0);
                            double epoch_bandwidth_DPU_CPU = (double) (0);

                            double epoch_time_model_average = (double) (0);
                            double epoch_bytes_model_average = (double) (0);
                            

                            // ITERATION AT DPU
                            fprintf(fp_dpu, "Run program on DPUs\n"); 
                            for(uint64_t rep = 0; rep < epochs; ++rep) { 
                                printf("Global_epoch: %lu\n", rep);
                                i = 0; 
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, results + NR_DPUS * NR_TASKLETS * rep + NR_TASKLETS * i)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "BENCHMARK", \
                                    0, \
                                    NR_TASKLETS*sizeof(dpu_result_t), DPU_XFER_DEFAULT)); 

                                
                                clock_gettime(CLOCK_MONOTONIC, &start);
                                // Copy W_ADMM_local
                                i = 0; 
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, W_ADMM_local + i * n_size)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                    max_rows_per_dpu * n_size_pad * sizeof(U) + communicate_Y_pad * sizeof(U), \
                                    n_size_pad_W_fp * sizeof(S), DPU_XFER_DEFAULT)); 
                                
                                // Copy U_Z_ADMM_local
                                i = 0;
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, U_Z_ADMM_local + i * n_size)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                    max_rows_per_dpu * n_size_pad * sizeof(U) + communicate_Y_pad * sizeof(U) + n_size_pad_W_fp * sizeof(S), \
                                    n_size_pad_W_fp * sizeof(S), DPU_XFER_DEFAULT)); 
                                
                                // Copy bias_W_ADMM_local
                                i = 0;
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, bias_W_ADMM_local + i)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                    max_rows_per_dpu * n_size_pad * sizeof(U) + communicate_Y_pad * sizeof(U) + 2*n_size_pad_W_fp * sizeof(S), \
                                    sizeof(LS), DPU_XFER_DEFAULT)); 
                                
                                clock_gettime(CLOCK_MONOTONIC, &end);
                                elapsed_time = (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                fprintf(fp_dpu, "Epoch %lu. Loading W_ADMM_global to DPUs. Elapsed time is %.9f s\n", rep, elapsed_time);

                                epoch_time_CPU_DPU += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                epoch_bytes_CPU_DPU += (double) (((double) (NR_DPUS)) * ((double) (2 * n_size_pad_W_fp * sizeof(S) + sizeof(LS))));


                                
                                // Launch kernel 
                                clock_gettime(CLOCK_MONOTONIC, &start);
                                DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS)); 
                                clock_gettime(CLOCK_MONOTONIC, &end);
                                elapsed_time = (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                fprintf(fp_dpu, "Epoch %lu. DPU kernel time. Elapsed time is %.9f s\n", rep, elapsed_time);


                                
                                
                                // Retrieve result
                                clock_gettime(CLOCK_MONOTONIC, &start);
                                // Retrieve W_ADMM_local
                                i = 0; 
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, W_ADMM_local + i * n_size)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                    max_rows_per_dpu * n_size_pad * sizeof(U) + communicate_Y_pad * sizeof(U), \
                                    n_size_pad_W_fp * sizeof(S), DPU_XFER_DEFAULT)); 
                                
                                // Retrieve bias_W_ADMM_local
                                i = 0;
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, bias_W_ADMM_local + i)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                    max_rows_per_dpu * n_size_pad * sizeof(U) + communicate_Y_pad * sizeof(U) + 2*n_size_pad_W_fp * sizeof(S), \
                                    sizeof(LS), DPU_XFER_DEFAULT)); 
                                
                                clock_gettime(CLOCK_MONOTONIC, &end);
                                elapsed_time = (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                fprintf(fp_dpu, "Epoch %lu. Retrieve the models of all the DPUs. Elapsed time is %.9f s\n", rep, elapsed_time);

                                epoch_time_DPU_CPU += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                epoch_bytes_DPU_CPU += (double) (((double) (NR_DPUS)) * ((double) (n_size_pad_W_fp * sizeof(S) + sizeof(LS))));
        


                                // Compute gradient
                                clock_gettime(CLOCK_MONOTONIC, &start);
                                memset(W_ADMM_global_buffer, 0, n_size*sizeof(LS));
                                memset(U_ADMM_global_buffer, 0, n_size*sizeof(LS));
                                memset(bias_W_ADMM_global_buffer, 0, sizeof(LS));
                                memset(bias_U_ADMM_global_buffer, 0, sizeof(LS));
                                pthread_t model_average_worker_thread[17];
                                for (uint32_t s = 0; s < 17; ++s) {
                                    int32_t thread_return_value = pthread_create(&model_average_worker_thread[s], NULL, model_average_function, (void *) &thread_info_model_average_parallel_list[s]);
                                    if (thread_return_value) {
                                        printf("Failed to create thread, %d\n", thread_return_value);
                                    }
                                }
                                for (uint32_t s = 0; s < 17; ++s) {
                                    pthread_join(model_average_worker_thread[s], NULL);
                                }


                                
                                
                                clock_gettime(CLOCK_MONOTONIC, &end);
                                elapsed_time = (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                fprintf(fp_dpu, "Epoch %lu. Model averaging on CPU. Elapsed time is %.9f s\n", rep, elapsed_time);

                                epoch_time_model_average += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                epoch_bytes_model_average += (double) (((double) (NR_DPUS)) * ((double) (3 * n_size_pad_W_fp * sizeof(S) + sizeof(LS))));


                                


                                for (uint32_t l = 0; l < n_size; ++l) {
                                    W_ADMM_global_float_error[l] = (float) W_ADMM_global[l] / CAST;
                                    
                                    W_ADMM_checkpoint_for_compute_error[rep*n_size + l] = W_ADMM_global_float_error[l];
                                }
                                *bias_W_ADMM_dpu_float_error = (float) *bias_W_ADMM_global / CAST;
                                bias_W_ADMM_checkpoint_for_compute_error[rep*1] = *bias_W_ADMM_dpu_float_error;
                                
                                for (uint32_t l = 0; l < n_size; ++l) {
                                    Z_ADMM_global_float_error[l] = (float) Z_ADMM_global[l] / CAST;
                                    
                                    Z_ADMM_checkpoint_for_compute_error[rep*n_size + l] = Z_ADMM_global_float_error[l];
                                }
                                *bias_Z_ADMM_dpu_float_error = (float) *bias_Z_ADMM_global / CAST;
                                bias_Z_ADMM_checkpoint_for_compute_error[rep*1] = *bias_Z_ADMM_dpu_float_error;

                                i = 0; 
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, results + NR_DPUS * NR_TASKLETS * rep + NR_TASKLETS * i)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "BENCHMARK", \
                                    0, \
                                    NR_TASKLETS*sizeof(dpu_result_t), DPU_XFER_DEFAULT)); 
                                
                                for (uint64_t m = 0; m < NR_DPUS; ++m) {
                                    for (uint64_t x = 0; x < NR_TASKLETS; ++x) {
                                        results_double[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].time_init = (double) (((double) (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_init))/clock_speed_DPU);
                                        results_double[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].time_compute = (double) (((double) (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_compute))/clock_speed_DPU);
                                        results_double[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].time_mram_to_wram = (double) (((double) (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_mram_to_wram))/clock_speed_DPU);
                                        results_double[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_mram_to_wram = (uint64_t) (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_mram_to_wram);
                                        results_double[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].time_wram_to_mram = (double) (((double) (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_wram_to_mram))/clock_speed_DPU);
                                        results_double[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_wram_to_mram = (uint64_t) (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_wram_to_mram); 
                                        
                                        if (m == 0 && x < 3) {
                                            printf("DPU_0: compute_time_mram_to_wram = %.9f\n", results_double[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].time_mram_to_wram);
                                            printf("DPU_0: compute_time_wram_to_mram = %.9f\n", results_double[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].time_wram_to_mram);
                                        }
                                        
                                        if (m == 0 && x < 3) {
                                            printf("DPU_0: bytes_mram_to_wram = %lu\n", results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_mram_to_wram);
                                            printf("DPU_0: bytes_wram_to_mram = %lu\n", results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_wram_to_mram);
                                        }
                                        
                                        

                                    }
                                }
                                

                            } // iter end 

                            

                            printf("End train model_global\n");
                            fprintf(fp_dpu,"\n\nBenchmark:communications_init = %lu, communications_per_epoch = %lu, communications_CPU_DPU_per_epoch = %lu, communications_DPU_CPU_per_epoch = %lu\n\n",communications_init,communications_per_epoch,communications_CPU_DPU_per_epoch,communications_DPU_CPU_per_epoch);


                            epoch_time_CPU_DPU = (double) (epoch_time_CPU_DPU/epochs_double);
                            epoch_bytes_CPU_DPU = (double) (epoch_bytes_CPU_DPU/epochs_double);
                            epoch_bytes_CPU_DPU = (double) (epoch_bytes_CPU_DPU/((double) (1e9)));
                            
                            epoch_time_DPU_CPU = (double) (epoch_time_DPU_CPU/epochs_double);
                            epoch_bytes_DPU_CPU = (double) (epoch_bytes_DPU_CPU/epochs_double);
                            epoch_bytes_DPU_CPU = (double) (epoch_bytes_DPU_CPU/((double) (1e9)));

                            double total_epoch_time_CPU_and_DPU_combined = (double) (epoch_time_CPU_DPU+epoch_time_DPU_CPU);
                            double total_epoch_bytes_CPU_and_DPU_combined = (double) (epoch_bytes_CPU_DPU+epoch_bytes_DPU_CPU);

                            double total_epoch_bandwidth_CPU_and_DPU_combined = (double) (total_epoch_bytes_CPU_and_DPU_combined/total_epoch_time_CPU_and_DPU_combined);
                            
                            fprintf(fp_dpu, "Benchmark:CPU_and_DPU_combined_epoch_bandwidth = %.9f GB/s, CPU_and_DPU_combined_epoch_elapsed_time = %.9f s, CPU_and_DPU_combined_epoch_giga_bytes = %.9f GB\n\n", total_epoch_bandwidth_CPU_and_DPU_combined,total_epoch_time_CPU_and_DPU_combined,total_epoch_bytes_CPU_and_DPU_combined);
                            
                            
                            epoch_bandwidth_CPU_DPU = (double) (epoch_bytes_CPU_DPU/epoch_time_CPU_DPU);
                            fprintf(fp_dpu, "Benchmark:CPU_DPU_epoch_bandwidth = %.9f GB/s, CPU_DPU_epoch_elapsed_time = %.9f s, CPU_DPU_epoch_giga_bytes = %.9f GB\n\n", epoch_bandwidth_CPU_DPU,epoch_time_CPU_DPU,epoch_bytes_CPU_DPU);


                            epoch_bandwidth_DPU_CPU = (double) (epoch_bytes_DPU_CPU/epoch_time_DPU_CPU);
                            fprintf(fp_dpu, "Benchmark:DPU_CPU_epoch_bandwidth = %.9f GB/s, DPU_CPU_epoch_elapsed_time = %.9f s, DPU_CPU_epoch_giga_bytes = %.9f GB\n\n", epoch_bandwidth_DPU_CPU,epoch_time_DPU_CPU,epoch_bytes_DPU_CPU);


                            epoch_time_model_average = (double) (epoch_time_model_average/epochs_double);
                            epoch_bytes_model_average = (double) (epoch_bytes_model_average/epochs_double);
                            epoch_bytes_model_average = (double) (epoch_bytes_model_average/((double) (1e9)));
                            fprintf(fp_dpu, "Benchmark:epoch_time_model_average = %.9f s, epoch_giga_bytes_model_average = %.9f GB\n\n\n\n", epoch_time_model_average, epoch_bytes_model_average);

                            
                            dpu_result_per_epoch_t* final_result_per_epoch = (dpu_result_per_epoch_t*) calloc(epochs, sizeof(dpu_result_per_epoch_t));
                            dpu_final_result_double_t* final_result = (dpu_final_result_double_t*) calloc(2,sizeof(dpu_final_result_double_t));
                            for (uint64_t rep=0; rep < epochs; ++rep) {
                                uint64_t tmp_max_time_init = (uint64_t) (0);
                                uint64_t tmp_max_time_compute = (uint64_t) (0);
                                uint64_t tmp_max_time_mram_to_wram = (uint64_t) (0);
                                uint64_t tmp_max_time_wram_to_mram = (uint64_t) (0);
                                for (uint64_t m = 0; m < NR_DPUS; ++m) {
                                    double tmp_bytes_mram_to_wram = (double) (0);
                                    double tmp_bytes_wram_to_mram = (double) (0); 
                                    for (uint64_t x = 0; x < NR_TASKLETS; ++x) {
                                        if (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_init > tmp_max_time_init) {
                                            tmp_max_time_init = results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_init;
                                        }
                                        if (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_compute > tmp_max_time_compute) {
                                            tmp_max_time_compute = results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_compute;
                                        }
                                        if (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_mram_to_wram > tmp_max_time_mram_to_wram) {
                                            tmp_max_time_mram_to_wram = results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_mram_to_wram;
                                        }
                                        if (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_wram_to_mram > tmp_max_time_wram_to_mram) {
                                            tmp_max_time_wram_to_mram = results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_wram_to_mram;
                                        }
                                        
                                        tmp_bytes_mram_to_wram += (double) (((double) (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_mram_to_wram))/((double) (1e9)));
                                        
                                        tmp_bytes_wram_to_mram += (double) (((double) (results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_wram_to_mram))/((double) (1e9)));
                                    }
                                    final_result_per_epoch[rep].bytes_mram_to_wram += tmp_bytes_mram_to_wram; 
                                    final_result_per_epoch[rep].bytes_wram_to_mram += tmp_bytes_wram_to_mram;
                                    
                                }
                                final_result_per_epoch[rep].cycles_init = tmp_max_time_init;
                                final_result_per_epoch[rep].cycles_compute = tmp_max_time_compute;
                                final_result_per_epoch[rep].cycles_mram_to_wram = tmp_max_time_mram_to_wram;
                                final_result_per_epoch[rep].cycles_wram_to_mram = tmp_max_time_wram_to_mram;
                                
                            }
                            
                            

                            double all_epochs_combined_bytes_mram_to_wram = (double) (0);
                            double all_epochs_combined_bytes_wram_to_mram = (double) (0);
                            uint64_t all_epochs_combined_cycles_init = (uint64_t) (0);
                            uint64_t all_epochs_combined_cycles_compute = (uint64_t) (0);
                            uint64_t all_epochs_combined_cycles_mram_to_wram = (uint64_t) (0);
                            uint64_t all_epochs_combined_cycles_wram_to_mram = (uint64_t) (0);
                            for (uint64_t rep = 0; rep < epochs; ++rep) {
                                all_epochs_combined_bytes_mram_to_wram += final_result_per_epoch[rep].bytes_mram_to_wram;
                                all_epochs_combined_bytes_wram_to_mram += final_result_per_epoch[rep].bytes_wram_to_mram;
                                all_epochs_combined_cycles_init += final_result_per_epoch[rep].cycles_init;
                                all_epochs_combined_cycles_compute += final_result_per_epoch[rep].cycles_compute;
                                all_epochs_combined_cycles_mram_to_wram += final_result_per_epoch[rep].cycles_mram_to_wram;
                                all_epochs_combined_cycles_wram_to_mram += final_result_per_epoch[rep].cycles_wram_to_mram;
                            }   
                            

                            final_result[0].giga_bytes_mram_to_wram = (double) (((double) (all_epochs_combined_bytes_mram_to_wram))/(epochs_double));
                            final_result[0].giga_bytes_wram_to_mram = (double) (((double) (all_epochs_combined_bytes_wram_to_mram))/(epochs_double));

                            final_result[0].time_init = (double) (((double) (all_epochs_combined_cycles_init))/(clock_speed_DPU*epochs_double));
                            final_result[0].time_compute = (double) (((double) (all_epochs_combined_cycles_compute))/(clock_speed_DPU*epochs_double));
                            final_result[0].time_mram_to_wram = (double) (((double) (all_epochs_combined_cycles_mram_to_wram))/(clock_speed_DPU*epochs_double));
                            final_result[0].time_wram_to_mram = (double) (((double) (all_epochs_combined_cycles_wram_to_mram))/(clock_speed_DPU*epochs_double));

                            
                            
                            fprintf(fp_dpu,"Benchmark:DPU_epoch_initialization_time = %.9f s\n\n",final_result[0].time_init);
                            fprintf(fp_dpu,"Benchmark:DPU_epoch_compute_time = %.9f s\n\n\n\n",final_result[0].time_compute);
                            

                            double dpu_final_mram_and_wram_combined_time = (double) (final_result[0].time_mram_to_wram + final_result[0].time_wram_to_mram);
                            double dpu_final_mram_and_wram_combined_giga_bytes = (double) (final_result[0].giga_bytes_mram_to_wram + final_result[0].giga_bytes_wram_to_mram);

                            double dpu_final_mram_and_wram_combined_bandwidth = (double) (dpu_final_mram_and_wram_combined_giga_bytes/dpu_final_mram_and_wram_combined_time);
                            
                            fprintf(fp_dpu, "Benchmark:mram_and_wram_combined_epoch_bandwidth = %.9f GB/s, mram_and_wram_combined_epoch_elapsed_time = %.9f s, mram_and_wram_combined_epoch_giga_bytes = %.9f GB\n\n", dpu_final_mram_and_wram_combined_bandwidth,dpu_final_mram_and_wram_combined_time,dpu_final_mram_and_wram_combined_giga_bytes);



                            double dpu_final_mram_to_wram_bandwidth = (double) (final_result[0].giga_bytes_mram_to_wram/final_result[0].time_mram_to_wram);
                            
                            fprintf(fp_dpu, "Benchmark:mram_to_wram_epoch_bandwidth = %.9f GB/s, mram_to_wram_epoch_elapsed_time = %.9f s, mram_to_wram_epoch_giga_bytes = %.9f GB\n\n", dpu_final_mram_to_wram_bandwidth, final_result[0].time_mram_to_wram, final_result[0].giga_bytes_mram_to_wram);
                            



                            double dpu_final_wram_to_mram_bandwidth = (double) (final_result[0].giga_bytes_wram_to_mram/final_result[0].time_wram_to_mram);
                            
                            fprintf(fp_dpu, "Benchmark:wram_to_mram_epoch_bandwidth = %.9f GB/s, wram_to_mram_epoch_elapsed_time = %.9f s, wram_to_mram_epoch_giga_bytes = %.9f GB\n\n", dpu_final_wram_to_mram_bandwidth, final_result[0].time_wram_to_mram, final_result[0].giga_bytes_wram_to_mram);
                            


                            

                            // Deallocation
                            free(input_args); 
                            free(dpu_info);
                            free(W_ADMM_global);  
                            free(U_ADMM_global);
                            free(W_ADMM_local);
                            free(Z_ADMM_global);
                            free(U_ADMM_local);
                            free(U_Z_ADMM_local);

                            free(bias_W_ADMM_global);
                            free(bias_U_ADMM_global);
                            free(bias_W_ADMM_local);
                            free(bias_Z_ADMM_global);
                            free(bias_U_ADMM_local);
                            free(bias_U_Z_ADMM_local);

                            free(results);
                            free(final_result_per_epoch);
                            free(final_result);
                            

                            fprintf(fp_dpu, "Experiment successfully completed.\n");
                            fclose(fp_dpu);

                            
                        }
                    }
                }
            }
        }
    }

    free(W_ADMM_global_buffer);
    free(U_ADMM_global_buffer);
    free(bias_W_ADMM_global_buffer);
    free(bias_U_ADMM_global_buffer);
    
    
    free(error_rate);
    free(reduction);
    free(sum_of_Y);
    free(hinge_loss);
    free(thread_info_compute_error_rate_list);
    free(thread_info_compute_loss_list);
    free(W_ADMM_checkpoint_for_compute_error);
    free(Z_ADMM_checkpoint_for_compute_error);
    free(bias_W_ADMM_checkpoint_for_compute_error);
    free(bias_Z_ADMM_checkpoint_for_compute_error);
    free(compute_error_rate_worker_list);
    free(reduction_worker_list);
    free(sum_of_Y_worker_list);
    free(compute_loss_worker_list);
    free(W_ADMM_global_float_error);
    free(Z_ADMM_global_float_error);
    free(bias_W_ADMM_dpu_float_error);
    free(bias_Z_ADMM_dpu_float_error);
    free(filename_experiment_dpu);
    free(X);
    free(Y);


    DPU_ASSERT(dpu_free(dpu_set));
    
    
    return 0;
    
}
