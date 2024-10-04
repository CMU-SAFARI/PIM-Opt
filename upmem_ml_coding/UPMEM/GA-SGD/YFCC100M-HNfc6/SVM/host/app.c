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


// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

static U *X;
static U *Y;

static U *X_test;
static U *Y_test;


void* gradient_average_function(void* thread_info_list){
    gradient_average_parallel_t* thread_info_list_current = (gradient_average_parallel_t*) thread_info_list;
    uint64_t thread_id = thread_info_list_current->thread_id;
    LS overflow_test = thread_info_list_current->overflow_test;
    LS underflow_test = thread_info_list_current->underflow_test;
    uint32_t b_size_frac_log = thread_info_list_current->b_size_frac_log;
    uint32_t learning_rate_plus_reg_term = thread_info_list_current->learning_rate_plus_reg_term;

    if (thread_id < 16) { 
        S* W_local_gradients_buffer = thread_info_list_current->W_local_gradients_buffer;
        LS* W_global_gradient_buffer = thread_info_list_current->W_global_gradient_buffer;

        S* W_dpu_fp = thread_info_list_current->W_dpu_fp;
        uint64_t number_of_features_per_thread = thread_info_list_current->number_of_features_per_thread;
        uint64_t n_size = thread_info_list_current->n_size;

        for (uint64_t m = 0; m < NR_DPUS; ++m) {
            for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
                W_global_gradient_buffer[x] += (LS) W_local_gradients_buffer[m*n_size + x];
            }
        }

        for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
            LS check_tmp = (LS) (W_global_gradient_buffer[x] >> b_size_frac_log);
            LS check_tmp_reg = (LS) (((LS) (W_dpu_fp[x])) >> learning_rate_plus_reg_term);
            LS gradient_update = (LS) (0);
            if (check_tmp != -1) {
                if (check_tmp_reg != -1) {
                    gradient_update = (LS) (check_tmp + check_tmp_reg);
                } else {
                    gradient_update = check_tmp;
                }
            } else {
                if (check_tmp_reg != -1) {
                    gradient_update = check_tmp_reg;
                }
            }
            LS tmp = (LS) (((LS) (W_dpu_fp[x])) - gradient_update); 
            if (tmp > overflow_test || tmp < underflow_test) {
                if (tmp > overflow_test) {
                    W_dpu_fp[x] = (S) (overflow_test - 1);
                } else {
                    W_dpu_fp[x] = (S) (underflow_test + 1);
                }
            } else {
                W_dpu_fp[x] = (S) (tmp);
            }
        } 
    } else {
        LS* bias_local_gradients_buffer = thread_info_list_current->bias_local_gradients_buffer;
        LS* bias_global_gradient_buffer = thread_info_list_current->bias_global_gradient_buffer;
        
        LS* bias_W_dpu_fp = thread_info_list_current->bias_W_dpu_fp;
        for (uint64_t m = 0; m < NR_DPUS; ++m) {
            *bias_global_gradient_buffer += bias_local_gradients_buffer[m];
        }
        LS gradient_update = (LS) (0);
        LS bias_global_gradient_buffer_tmp = *bias_global_gradient_buffer;
        LS check_tmp = (LS) (bias_global_gradient_buffer_tmp >> b_size_frac_log);
        if (check_tmp != -1) {
            gradient_update = check_tmp;
        }
        LS tmp = (LS) (*bias_W_dpu_fp - gradient_update);
        if (tmp > overflow_test || tmp < underflow_test) {
            if (tmp > overflow_test) {
                *bias_W_dpu_fp = (LS) (overflow_test - 1);
            } else {
                *bias_W_dpu_fp = (LS) (underflow_test + 1);
            }
        } else {
            *bias_W_dpu_fp = tmp;
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
    float* bias_W_host_float = thread_info_list_current->bias_W_host_float;
    uint64_t m_size = thread_info_list_current->m_size;
    uint64_t n_size = thread_info_list_current->n_size;

    for (uint64_t m = 0; m < m_size; ++m) {
        float dot_product = 0.0;
        for (uint64_t n = 0; n < n_size; ++n) {
            dot_product += ((float) X[m*n_size + n] / CAST)*W_host_float[n]; 
        }
        dot_product += *bias_W_host_float;
        
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
    float* bias_W_host_float = thread_info_list_current->bias_W_host_float;
    uint64_t m_size = thread_info_list_current->m_size;
    uint64_t n_size = thread_info_list_current->n_size;

    for (uint64_t m = 0; m < m_size; ++m) {
        double dot_product = 0.0;
        for (uint64_t n = 0; n < n_size; ++n) {
            dot_product += (double) (((float) X[m*n_size + n] / CAST)*W_host_float[n]);
        }
        double bias_W_host_float_tmp = (double) (*bias_W_host_float);
        dot_product += bias_W_host_float_tmp;
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

    uint64_t strong_scaling = SCALING;


    uint32_t array_b_size_frac[] = {4096, 8192, 16384, 32768};
    uint32_t array_b_size_frac_length = sizeof(array_b_size_frac) / sizeof(array_b_size_frac[0]);

    if (AE == 1) { // reproducing the results in our paper
        if (strong_scaling == 0) {
            if (NR_DPUS == 256 || NR_DPUS == 512 || NR_DPUS == 1024) {
                array_b_size_frac[0] = 8192;
                array_b_size_frac_length = 1;
            }
        }
        if (strong_scaling == 1) {
            array_b_size_frac[0] = 8192;
            array_b_size_frac_length = 1;
        }
    }

    uint32_t array_learning_rate[] = {6};
    uint32_t array_learning_rate_length = sizeof(array_learning_rate) / sizeof(array_learning_rate[0]);
    
    uint32_t array_reg_term[] = {8};
    uint32_t array_reg_term_length = sizeof(array_reg_term) / sizeof(array_reg_term[0]);

    uint32_t array_epochs[] = {10};
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
    read_test_input_yfcc100m(bufferX_test, bufferY_test, number_of_test_samples, strong_scaling);
    printf("\nBefore reading train input\n");
    read_input_yfcc100m(bufferX, bufferY, m_size, strong_scaling);
    
    
    printf("Before starting the experiments\n");
    char* filename_experiment_dpu = calloc(1000, sizeof(char));
    float* W_checkpoint_for_compute_error = calloc(max_epochs_length*n_size, sizeof(float));
    float* bias_checkpoint_for_compute_error = calloc(max_epochs_length*1, sizeof(float));
    float* compute_error_rate_worker_list = calloc(64, sizeof(float));
    uint64_t* reduction_worker_list = calloc(64, sizeof(uint64_t));
    uint64_t* sum_of_Y_worker_list = calloc(64, sizeof(uint64_t));
    double* compute_loss_worker_list = calloc(64, sizeof(double));
    float* W_dpu_float_error = calloc(n_size, sizeof(float));
    float* bias_dpu_float_error = calloc(1, sizeof(float));
    S* W_local_gradients_buffer = (S*) calloc(NR_DPUS*n_size, sizeof(S));
    LS* W_global_gradient_buffer = (LS*) calloc(n_size, sizeof(LS));
    LS* bias_local_gradients_buffer = (LS*) calloc(NR_DPUS, sizeof(LS));
    LS* bias_global_gradient_buffer = (LS*) calloc(1, sizeof(LS));
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
                    for (uint32_t le = 0; le < array_task_epochs_length; ++le) {
                        uint32_t task_epochs = array_task_epochs[le];
                        printf("Initialized task_epochs\n");
                        uint32_t nr_batches = (uint32_t) m_size/(b_size_frac);
                        
                        memset(filename_experiment_dpu, 0, 1000*sizeof(char));
                        memset(W_checkpoint_for_compute_error, 0, max_epochs_length*n_size*sizeof(float));
                        memset(bias_checkpoint_for_compute_error, 0, max_epochs_length*1*sizeof(float));
                        memset(W_dpu_float_error, 0, n_size*sizeof(float));
                        memset(bias_dpu_float_error, 0, 1*sizeof(float));
                        memset(W_local_gradients_buffer, 0, NR_DPUS*n_size*sizeof(S));
                        memset(bias_local_gradients_buffer, 0, NR_DPUS*sizeof(LS));
                        memset(bias_global_gradient_buffer, 0, 1*sizeof(LS));
                        memset(W_global_gradient_buffer, 0, n_size*sizeof(LS));
                        memset(thread_info_compute_error_rate_list, 0, 64*sizeof(compute_error_rate_worker_t));
                        memset(thread_info_compute_loss_list, 0, 64*sizeof(compute_loss_worker_t));
                        memset(error_rate, 0, sizeof(float));
                        memset(reduction, 0, sizeof(uint64_t));
                        memset(sum_of_Y, 0, sizeof(uint64_t));
                        memset(hinge_loss, 0, sizeof(double));

                        
                        printf("About to create fdpu file\n");

                        if (strong_scaling == 0) {
                            sprintf(filename_experiment_dpu, "%s/results_UPMEM/YFCC100M-HNfc6/weak_scaling/ga_SVM_uint32/svm_reg_DPU_arch__yfcc100m__uint32__GA__NR_DPUS_%u__NR_TASKLETS_%u__m_size_%lu__m_test_size_%lu__n_size_%u__b_size_frac_%u__nr_batches_%u__learning_rate_%u__reg_term_%u__epochs_%u__task_epochs_%u.txt", DEST_DIR, NR_DPUS, NR_TASKLETS, m_size, number_of_test_samples, n_size, b_size_frac, nr_batches, learning_rate, reg_term, epochs, task_epochs);
                        } else {
                            sprintf(filename_experiment_dpu, "%s/results_UPMEM/YFCC100M-HNfc6/strong_scaling/ga_SVM_uint32/svm_reg_DPU_arch__yfcc100m__uint32__GA__NR_DPUS_%u__NR_TASKLETS_%u__m_size_%lu__m_test_size_%lu__n_size_%u__b_size_frac_%u__nr_batches_%u__learning_rate_%u__reg_term_%u__epochs_%u__task_epochs_%u.txt", DEST_DIR, NR_DPUS, NR_TASKLETS, m_size, number_of_test_samples, n_size, b_size_frac, nr_batches, learning_rate, reg_term, epochs, task_epochs);
                        }

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

                        uint32_t b_size_frac_DPU = (uint32_t) (b_size_frac/nr_of_dpus);
                        
                        uint32_t batch_size_frac_Y_pad = (b_size_frac*sizeof(U) % 8) == 0 ? b_size_frac : roundup(b_size_frac, (8/sizeof(U)));

                        uint32_t batch_size_frac_Y_pad_DPU = (b_size_frac_DPU*sizeof(U) % 8) == 0 ? b_size_frac_DPU : roundup(b_size_frac_DPU, (8/sizeof(U)));

                        uint32_t size_of_batch = b_size_frac * nr_of_dpus;
                        uint32_t b_size_frac_log = (U) abs((int32_t)log2(b_size_frac));
                        

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

                            input_args[i].batch_size_frac_Y_pad = batch_size_frac_Y_pad_DPU;

                            input_args[i].nr_batches = nr_batches;
                            input_args[i].learning_rate = learning_rate; 
                            input_args[i].current_batch_id = (U) (0);
                            input_args[i].b_size_frac = b_size_frac_DPU;

                            input_args[i].global_epoch_current = (uint32_t) (0);

                            input_args[i].samples_loop_transfer = samples_loop_transfer;
                            input_args[i].transfer = transfer;
                            input_args[i].samples_per_transfer = samples_per_transfer;

                            

                        }
                        

                        // init Weight for DPU 
                        S* W_dpu_fp = calloc(n_size_pad_W_fp, sizeof(S)); 
                        LS* bias_W_dpu_fp = calloc(1, sizeof(LS));

                        LS overflow_test = (LS) CAST;
                        LS underflow_test = -overflow_test;

                        // gradient average with 16 threads
                        gradient_average_parallel_t* thread_info_gradient_average_parallel_list = (gradient_average_parallel_t*) calloc(17, sizeof(gradient_average_parallel_t));
                        uint64_t start_index_thread_model_average = (uint64_t) (256);
                        for (uint64_t s = 0; s < 17; ++s) {
                            thread_info_gradient_average_parallel_list[s].thread_id = s;
                            thread_info_gradient_average_parallel_list[s].overflow_test = overflow_test;
                            thread_info_gradient_average_parallel_list[s].underflow_test = underflow_test;

                            thread_info_gradient_average_parallel_list[s].b_size_frac_log = b_size_frac_log;
                            thread_info_gradient_average_parallel_list[s].learning_rate_plus_reg_term = (uint32_t) (learning_rate + reg_term);

                            if (s < 16) {
                                thread_info_gradient_average_parallel_list[s].W_local_gradients_buffer = W_local_gradients_buffer + s * start_index_thread_model_average;
                                thread_info_gradient_average_parallel_list[s].W_global_gradient_buffer = W_global_gradient_buffer + s * start_index_thread_model_average;

                                thread_info_gradient_average_parallel_list[s].W_dpu_fp = W_dpu_fp + s * start_index_thread_model_average;
                                thread_info_gradient_average_parallel_list[s].number_of_features_per_thread = (uint64_t) (256);
                                thread_info_gradient_average_parallel_list[s].n_size = n_size_pad_W_fp;
                            } else {
                                thread_info_gradient_average_parallel_list[s].bias_local_gradients_buffer = bias_local_gradients_buffer;
                                thread_info_gradient_average_parallel_list[s].bias_global_gradient_buffer = bias_global_gradient_buffer;
                                thread_info_gradient_average_parallel_list[s].bias_W_dpu_fp = bias_W_dpu_fp;
                            }
                            
                            
                            
                        }
                        
                        clock_gettime(CLOCK_MONOTONIC, &end);
                        elapsed_time = (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                        fprintf(fp_dpu, "Preprocessing and initialization of variables. Elapsed time is %.9f s\n", elapsed_time);
                        

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
                        
                        

                        // ITERATION AT DPU
                        fprintf(fp_dpu, "Run program on DPUs\n"); 
                        for(uint64_t rep = 0; rep < epochs; ++rep) { 
                            printf("Global_epoch: %lu\n", rep);
                            double elapsed_time_loading_to_DPUs = (double) (0);
                            double elapsed_time_DPU_kernel_time = (double) (0);
                            double elapsed_time_retrieving_results = (double) (0);
                            double elapsed_time_gradient_averaging = (double) (0);
                            for (uint32_t current_batch_id = 0; current_batch_id < nr_batches; ++current_batch_id) { 
                                // Copy W 
                                clock_gettime(CLOCK_MONOTONIC, &start);
                                i = 0; 
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, W_dpu_fp)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                    max_rows_per_dpu * n_size_pad * sizeof(U) + communicate_Y_pad * sizeof(U), \
                                    n_size_pad_W_fp * sizeof(S), DPU_XFER_DEFAULT)); 
                                i = 0; 
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, bias_W_dpu_fp)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                    max_rows_per_dpu * n_size_pad * sizeof(U) + communicate_Y_pad * sizeof(U) + n_size_pad_W_fp * sizeof(S), \
                                    sizeof(LS), DPU_XFER_DEFAULT)); 
                                

                                clock_gettime(CLOCK_MONOTONIC, &end);
                                elapsed_time_loading_to_DPUs += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                


                                
                                // Launch kernel 
                                clock_gettime(CLOCK_MONOTONIC, &start);
                                DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS)); 
                                clock_gettime(CLOCK_MONOTONIC, &end);
                                elapsed_time_DPU_kernel_time += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                


                           
                                
                                // Retrieve result
                                clock_gettime(CLOCK_MONOTONIC, &start);
                                i = 0;
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, W_local_gradients_buffer + i * n_size_pad_W_fp)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, n_size_pad_W_fp * sizeof(S), \
                                    DPU_XFER_DEFAULT)); 
                                
                                i = 0;
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, bias_local_gradients_buffer + i)); 
                                }
                                // We are storing bias_local_gradients_buffer on DPU side in bias_W_dpu_fp, next batch we are overwriting it again anyway
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                    max_rows_per_dpu * n_size_pad * sizeof(U) + communicate_Y_pad * sizeof(U) + n_size_pad_W_fp * sizeof(S), \
                                    sizeof(LS), DPU_XFER_DEFAULT)); 
                                
                                clock_gettime(CLOCK_MONOTONIC, &end);
                                elapsed_time_retrieving_results += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                
        


                                // Compute gradient
                                clock_gettime(CLOCK_MONOTONIC, &start);
                                memset(W_global_gradient_buffer, 0, n_size*sizeof(LS));
                                memset(bias_global_gradient_buffer, 0, sizeof(LS));
                                

                                
                                // start_index for each thread 16 of them 
                                pthread_t gradient_average_worker_thread[17];
                                for (uint32_t s = 0; s < 17; ++s) {
                                    int32_t thread_return_value = pthread_create(&gradient_average_worker_thread[s], NULL, gradient_average_function, (void *) &thread_info_gradient_average_parallel_list[s]);
                                    if (thread_return_value) {
                                        printf("Failed to create thread, %d\n", thread_return_value);
                                    }
                                }
                                for (uint32_t s = 0; s < 17; ++s) {
                                    pthread_join(gradient_average_worker_thread[s], NULL);
                                }

                                clock_gettime(CLOCK_MONOTONIC, &end);
                                elapsed_time_gradient_averaging += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                


                                
                            }
                            fprintf(fp_dpu, "Epoch %lu. Loading W_dpu_fp to DPUs. Elapsed time is %.9f s\n", rep, elapsed_time_loading_to_DPUs);
                            fprintf(fp_dpu, "Epoch %lu. DPU kernel time. Elapsed time is %.9f s\n", rep, elapsed_time_DPU_kernel_time);
                            fprintf(fp_dpu, "Epoch %lu. Retrieve the gradients of all the DPUs. Elapsed time is %.9f s\n", rep, elapsed_time_retrieving_results);
                            fprintf(fp_dpu, "Epoch %lu. Gradient averaging on CPU. Elapsed time is %.9f s\n", rep, elapsed_time_gradient_averaging);

                            printf("Epoch %lu. Averaged model W_dpu_fp.\n", rep);
                            for (uint32_t l = 0; l < n_size; ++l) {
                                W_dpu_float_error[l] = (float) W_dpu_fp[l] / CAST;
                                if (l < n_size-1) {
                                    printf("%.9f, ", W_dpu_float_error[l]);
                                } else {
                                    printf("%.9f\n", W_dpu_float_error[l]);
                                }
                                W_checkpoint_for_compute_error[rep*n_size + l] = W_dpu_float_error[l];
                            }
                            *bias_dpu_float_error = (float) *bias_W_dpu_fp / CAST;
                            bias_checkpoint_for_compute_error[rep*1] = *bias_dpu_float_error;
                            printf("Bias = %.9f\n", *bias_dpu_float_error);
                        

                        } // iter end 

                        

                        printf("End train model_global\n");
                        fprintf(fp_dpu, "Gathering data on compute accuracy and others\nThis is after the timing experiments have successfully been completed\n");
                        

                        for (uint64_t i = 0; i < epochs; ++i) { 
                            // Training error
                            fprintf(fp_dpu, "Epoch %lu. Compute training accuracy.\n", i);
                            for (uint64_t s = 0; s < 64; ++s) {
                                thread_info_compute_error_rate_list[s].W_host_float = W_checkpoint_for_compute_error+(i*n_size);
                                thread_info_compute_loss_list[s].W_host_float = W_checkpoint_for_compute_error+(i*n_size);
                                thread_info_compute_error_rate_list[s].bias_W_host_float = bias_checkpoint_for_compute_error + i;
                                thread_info_compute_loss_list[s].bias_W_host_float = bias_checkpoint_for_compute_error + i;
                            }
                            memset(error_rate, 0, sizeof(float));
                            memset(reduction, 0, sizeof(uint64_t));
                            memset(sum_of_Y, 0, sizeof(uint64_t));
                            memset(compute_error_rate_worker_list, 0, 64*sizeof(float));
                            memset(reduction_worker_list, 0, 64*sizeof(uint64_t));
                            memset(sum_of_Y_worker_list, 0, 64*sizeof(uint64_t));
                            for (uint64_t s = 0; s < 64; ++s) {
                                thread_info_compute_error_rate_list[s].error_rate = compute_error_rate_worker_list + s;
                                thread_info_compute_error_rate_list[s].reduction = reduction_worker_list + s;
                                thread_info_compute_error_rate_list[s].sum_of_Y = sum_of_Y_worker_list + s;
                                thread_info_compute_error_rate_list[s].n_size = (uint64_t) n_size;


                                thread_info_compute_error_rate_list[s].X = bufferX + s * start_index_worker_train * ((uint64_t) n_size);
                                thread_info_compute_error_rate_list[s].Y = bufferY + s * start_index_worker_train;
                                thread_info_compute_error_rate_list[s].m_size = start_index_worker_train;
                            }
                            pthread_t compute_train_error_worker_thread[64];
                            for (uint32_t s = 0; s < 64; ++s) {
                                int32_t thread_return_value = pthread_create(&compute_train_error_worker_thread[s], NULL, compute_error_rate_parallel, (void *) &thread_info_compute_error_rate_list[s]);
                                if (thread_return_value) {
                                    printf("Failed to create thread, %d\n", thread_return_value);
                                }
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                pthread_join(compute_train_error_worker_thread[s], NULL);
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                *reduction += reduction_worker_list[s];
                                *sum_of_Y += sum_of_Y_worker_list[s];
                            }
                            *error_rate = ((float) ((float) *reduction / ((uint64_t) m_size))*100);
                            float train_accuracy = (float) (((float) (100.0)) - *error_rate);

                            fprintf(fp_dpu, "Epoch %lu. Training accuracy of averaged model = %.5f, reduction = %lu, sum_of_Y = %lu\n", i, train_accuracy, *reduction, *sum_of_Y);
                            //Test error
                            memset(error_rate, 0, sizeof(float));
                            memset(reduction, 0, sizeof(uint64_t));
                            memset(sum_of_Y, 0, sizeof(uint64_t));
                            memset(compute_error_rate_worker_list, 0, 64*sizeof(float));
                            memset(reduction_worker_list, 0, 64*sizeof(uint64_t));
                            memset(sum_of_Y_worker_list, 0, 64*sizeof(uint64_t));
                            fprintf(fp_dpu, "Epoch %lu. Compute test accuracy.\n", i);
                            for (uint64_t s = 0; s < 64; ++s) {
                                thread_info_compute_error_rate_list[s].error_rate = compute_error_rate_worker_list + s;
                                thread_info_compute_error_rate_list[s].reduction = reduction_worker_list + s;
                                thread_info_compute_error_rate_list[s].sum_of_Y = sum_of_Y_worker_list + s;
                                thread_info_compute_error_rate_list[s].n_size = (uint64_t) n_size;


                                thread_info_compute_error_rate_list[s].X = bufferX_test + s * start_index_worker_test * ((uint64_t) n_size);
                                thread_info_compute_error_rate_list[s].Y = bufferY_test + s * start_index_worker_test;
                                thread_info_compute_error_rate_list[s].m_size = start_index_worker_test;
                                if ((m_size != (64*start_index_worker_test)) && (s == 63)) {
                                    thread_info_compute_error_rate_list[s].m_size = number_of_test_samples - (63 * start_index_worker_test);
                                }
                            }
                            pthread_t compute_test_error_worker_thread[64];
                            for (uint32_t s = 0; s < 64; ++s) {
                                int32_t thread_return_value = pthread_create(&compute_test_error_worker_thread[s], NULL, compute_error_rate_parallel, (void *) &thread_info_compute_error_rate_list[s]);
                                if (thread_return_value) {
                                    printf("Failed to create thread, %d\n", thread_return_value);
                                }
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                pthread_join(compute_test_error_worker_thread[s], NULL);
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                *reduction += reduction_worker_list[s];
                                *sum_of_Y += sum_of_Y_worker_list[s];
                            }
                            *error_rate = ((float) ((float) *reduction / ((uint64_t) number_of_test_samples))*100);
                            float test_accuracy = (float) (((float) (100.0)) - *error_rate);
                            
                            fprintf(fp_dpu, "Epoch %lu. Test accuracy of averaged model = %.5f, reduction = %lu, sum_of_Y = %lu\n", i, test_accuracy, *reduction, *sum_of_Y);

                            // Compute hinge loss
                            fprintf(fp_dpu, "Epoch %lu. Compute training hinge loss.\n", i);

                            memset(hinge_loss, 0, sizeof(double));
                            memset(compute_loss_worker_list, 0, 64*sizeof(double));

                            // calculate loss of regularization term
                            double w_squared = (double) (0);
                            for (uint32_t s = 0; s < n_size; ++s) {
                                w_squared += (double) (W_checkpoint_for_compute_error[(i*n_size) + s] * W_checkpoint_for_compute_error[(i*n_size) + s]);
                            }
                            // w_squared = sqrt(w_squared); --> we have squared so no sqrt necessary
                            double reg_term_double = (double) pow(2, reg_term+1); 
                            w_squared = w_squared / reg_term_double;

                            for (uint64_t s = 0; s < 64; ++s) {
                                thread_info_compute_loss_list[s].hinge_loss = compute_loss_worker_list + s;
                                thread_info_compute_loss_list[s].n_size = (uint64_t) n_size;

                                thread_info_compute_loss_list[s].X = bufferX + s * start_index_worker_train * ((uint64_t) n_size);
                                thread_info_compute_loss_list[s].Y = bufferY + s * start_index_worker_train;
                                thread_info_compute_loss_list[s].m_size = start_index_worker_train;
                            }
                            pthread_t compute_loss_worker_thread[64];
                            for (uint32_t s = 0; s < 64; ++s) {
                                int32_t thread_return_value = pthread_create(&compute_loss_worker_thread[s], NULL, compute_hinge_loss_parallel, (void *) &thread_info_compute_loss_list[s]);
                                if (thread_return_value) {
                                    printf("Failed to create thread, %d\n", thread_return_value);
                                }
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                pthread_join(compute_loss_worker_thread[s], NULL);
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                *hinge_loss += compute_loss_worker_list[s];
                            }
                            *hinge_loss = *hinge_loss / ((uint64_t) m_size);
                            double only_hinge_loss = *hinge_loss;
                            *hinge_loss += w_squared;
                            fprintf(fp_dpu, "Epoch %lu. Training hinge loss of averaged model = %.9f, only hinge loss = %.9f, only l2 loss = %.9f\n", i, *hinge_loss, only_hinge_loss, w_squared);
                            // Compute test cross entropy loss
                            memset(hinge_loss, 0, sizeof(double));
                            memset(compute_loss_worker_list, 0, 64*sizeof(double));
                            fprintf(fp_dpu, "Epoch %lu. Compute test hinge loss.\n", i);
                            for (uint64_t s = 0; s < 64; ++s) {
                                thread_info_compute_loss_list[s].hinge_loss = compute_loss_worker_list + s;
                                thread_info_compute_loss_list[s].n_size = (uint64_t) n_size;


                                thread_info_compute_loss_list[s].X = bufferX_test + s * start_index_worker_test * ((uint64_t) n_size);
                                thread_info_compute_loss_list[s].Y = bufferY_test + s * start_index_worker_test;
                                thread_info_compute_loss_list[s].m_size = start_index_worker_test;
                                if ((m_size != (64*start_index_worker_test)) && (s == 63)) {
                                    thread_info_compute_loss_list[s].m_size = number_of_test_samples - (63 * start_index_worker_test);
                                }
                            }
                            pthread_t compute_loss_test_worker_thread[64];
                            for (uint32_t s = 0; s < 64; ++s) {
                                int32_t thread_return_value = pthread_create(&compute_loss_test_worker_thread[s], NULL, compute_hinge_loss_parallel, (void *) &thread_info_compute_loss_list[s]);
                                if (thread_return_value) {
                                    printf("Failed to create thread, %d\n", thread_return_value);
                                }
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                pthread_join(compute_loss_test_worker_thread[s], NULL);
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                *hinge_loss += compute_loss_worker_list[s];
                            }
                            *hinge_loss = *hinge_loss / ((uint64_t) number_of_test_samples);
                            only_hinge_loss = *hinge_loss;
                            *hinge_loss += w_squared;
                            fprintf(fp_dpu, "Epoch %lu. Test hinge loss of averaged model = %.9f, only hinge loss = %.9f, only l2 loss = %.9f\n\n", i, *hinge_loss, only_hinge_loss, w_squared);
                        }

                        fprintf(fp_dpu, "\n\n\n\n\n\n\n\n\n\n");
                        fprintf(fp_dpu, "Printing complete models for each global epoch with bias at location 0\n");
                        for (uint64_t i = 0; i < epochs; ++i) { 
                            uint64_t start_print_index = (uint64_t) (i*n_size);
                            fprintf(fp_dpu, "Epoch %lu. Model W_dpu_fp.\n", i);
                            fprintf(fp_dpu,"%.9f,", bias_checkpoint_for_compute_error[i]);
                            for (uint64_t j = 0; j < n_size; ++j) {
                                if (j < n_size-1) {
                                    fprintf(fp_dpu,"%.9f,", W_checkpoint_for_compute_error[start_print_index + j]);
                                } else if (j == n_size-1) {
                                    fprintf(fp_dpu,"%.9f\n", W_checkpoint_for_compute_error[start_print_index + j]);
                                }
                            }
                        }


                        
                        

                        // Deallocation
                        free(input_args); 
                        free(dpu_info);
                        free(W_dpu_fp);  
                        

                        fprintf(fp_dpu, "Experiment successfully completed.\n");
                        fclose(fp_dpu);

                    }
                }
            }
        }
    }
    

    free(error_rate);
    free(reduction);
    free(sum_of_Y);
    free(hinge_loss);
    free(W_local_gradients_buffer);
    free(W_global_gradient_buffer);
    free(bias_local_gradients_buffer);
    free(bias_global_gradient_buffer);
    free(thread_info_compute_error_rate_list);
    free(thread_info_compute_loss_list);
    free(W_checkpoint_for_compute_error);
    free(bias_checkpoint_for_compute_error);
    free(compute_error_rate_worker_list);
    free(reduction_worker_list);
    free(sum_of_Y_worker_list);
    free(compute_loss_worker_list);
    free(W_dpu_float_error);
    free(bias_dpu_float_error);
    free(filename_experiment_dpu);
    free(X);
    free(Y);


    DPU_ASSERT(dpu_free(dpu_set));
    return 0;

    
}
