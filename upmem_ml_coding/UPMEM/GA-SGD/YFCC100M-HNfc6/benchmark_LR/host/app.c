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
        double sigmoid_temp = 1 / (1.0 + exp((double)(-dot_product))); 
        
        S predict_temp = sigmoid_temp >= 0.5 ? ((S) 1) : ((S) 0); // before 1:0
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

void* compute_cross_entropy_loss_parallel(void* thread_info_list) {
    compute_loss_worker_t* thread_info_list_current = (compute_loss_worker_t*) thread_info_list;
    double* cross_entropy_loss = thread_info_list_current->cross_entropy_loss;
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
        double sigmoid_tmp = 1 / (1.0 + exp((double)(-dot_product))); 
        
        double cross_entropy_loss_tmp = Y[m] == 1 ? log(sigmoid_tmp) : log(1 - sigmoid_tmp);
        *cross_entropy_loss += cross_entropy_loss_tmp;
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


    uint32_t array_b_size_frac[] = {4096, 8192, 16384, 32768};
    uint32_t array_b_size_frac_length = sizeof(array_b_size_frac) / sizeof(array_b_size_frac[0]);

    if (AE == 1) { // reproducing the results in our paper
        array_b_size_frac[0] = 4096;
        array_b_size_frac_length = 1;
    }


    uint32_t array_learning_rate[] = {4};
    uint32_t array_learning_rate_length = sizeof(array_learning_rate) / sizeof(array_learning_rate[0]);
    
    uint32_t array_reg_term[] = {10};
    uint32_t array_reg_term_length = sizeof(array_reg_term) / sizeof(array_reg_term[0]);

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

        thread_info_compute_loss_list[s].cross_entropy_loss = compute_loss_worker_list + s;
        thread_info_compute_loss_list[s].m_size = start_index_worker_train;
        thread_info_compute_loss_list[s].n_size = (uint64_t) n_size;
    }

    
    float* error_rate = calloc(1, sizeof(float));
    uint64_t* reduction = calloc(1, sizeof(uint64_t));
    uint64_t* sum_of_Y = calloc(1, sizeof(uint64_t));
    double* cross_entropy_loss = calloc(1, sizeof(double));

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
                        memset(cross_entropy_loss, 0, sizeof(double));


                        printf("About to create fdpu file\n");
                        sprintf(filename_experiment_dpu, "%s/benchmark_UPMEM/YFCC100M-HNfc6/ga_LR_uint32/lr_reg_DPU_arch__yfcc100m__uint32__GA__NR_DPUS_%u__NR_TASKLETS_%u__m_size_%lu__m_test_size_%lu__n_size_%u__b_size_frac_%u__nr_batches_%u__learning_rate_%u__reg_term_%u__epochs_%u__task_epochs_%u.txt", DEST_DIR, NR_DPUS, NR_TASKLETS, m_size, number_of_test_samples, n_size, b_size_frac, nr_batches, learning_rate, reg_term, epochs, task_epochs);

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
                        dpu_result_t* results = (dpu_result_t*) calloc(NR_TASKLETS*NR_DPUS*epochs, sizeof(dpu_result_t));
                        dpu_result_t* results_ga = (dpu_result_t*) calloc(NR_TASKLETS*NR_DPUS*epochs, sizeof(dpu_result_t));
                        uint64_t communications_init = (uint64_t) (4);
                        uint64_t communications_per_epoch = (uint64_t) (0);
                        uint64_t communications_CPU_DPU_per_epoch = (uint64_t) (0);
                        uint64_t communications_DPU_CPU_per_epoch = (uint64_t) (0);

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
                        
                        clock_gettime(CLOCK_MONOTONIC, &start);
                        broadcast_tables(dpu_set);
                        clock_gettime(CLOCK_MONOTONIC, &end);
                        elapsed_time = (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                        fprintf(fp_dpu, "TransPimLib: Create and broadcast tables to DPUs. Elapsed time is %.9f s\n", elapsed_time);
                        
                        time_initialization += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                        bytes_initialization += (double) (((uint64_t) (NR_DPUS)) * ((uint64_t)(1<<22)));
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
                                
                                epoch_time_CPU_DPU += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                epoch_bytes_CPU_DPU += (double) (((double) (NR_DPUS)) * ((double) (n_size_pad_W_fp * sizeof(S) + sizeof(LS))));
                                communications_CPU_DPU_per_epoch += (uint64_t) (1);


                                
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

                                epoch_time_DPU_CPU += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                epoch_bytes_DPU_CPU += (double) (((double) (NR_DPUS)) * ((double) (n_size_pad_W_fp * sizeof(S) + sizeof(LS))));
                                communications_DPU_CPU_per_epoch += (uint64_t) (1);
                                
        


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

                                epoch_time_model_average += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                epoch_bytes_model_average += (double) (((double) (NR_DPUS)) * ((double) (n_size_pad_W_fp * sizeof(S) + sizeof(LS))));

                                i = 0; 
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, results + NR_DPUS * NR_TASKLETS * rep + NR_TASKLETS * i)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "BENCHMARK", \
                                    0, \
                                    NR_TASKLETS*sizeof(dpu_result_t), DPU_XFER_DEFAULT)); 
                                
                                for (uint64_t m = 0; m < NR_DPUS; ++m) {
                                    for (uint64_t x = 0; x < NR_TASKLETS; ++x) {
                                        results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_init += results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_init;
                                        results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_compute += results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_compute;
                                        results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_mram_to_wram += results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_mram_to_wram;
                                        results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_mram_to_wram += results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_mram_to_wram;
                                        results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_wram_to_mram += results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_wram_to_mram;
                                        results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_wram_to_mram += results[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_wram_to_mram; 

                                    }
                                }

                            }
                            fprintf(fp_dpu, "Epoch %lu. Loading W_dpu_fp to DPUs. Elapsed time is %.9f s\n", rep, elapsed_time_loading_to_DPUs);
                            fprintf(fp_dpu, "Epoch %lu. DPU kernel time. Elapsed time is %.9f s\n", rep, elapsed_time_DPU_kernel_time);
                            fprintf(fp_dpu, "Epoch %lu. Retrieve the gradients of all the DPUs. Elapsed time is %.9f s\n", rep, elapsed_time_retrieving_results);
                            fprintf(fp_dpu, "Epoch %lu. Gradient averaging on CPU. Elapsed time is %.9f s\n", rep, elapsed_time_gradient_averaging);

                            for (uint32_t l = 0; l < n_size; ++l) {
                                W_dpu_float_error[l] = (float) W_dpu_fp[l] / CAST;
                                
                                W_checkpoint_for_compute_error[rep*n_size + l] = W_dpu_float_error[l];
                            }
                            *bias_dpu_float_error = (float) *bias_W_dpu_fp / CAST;
                            bias_checkpoint_for_compute_error[rep*1] = *bias_dpu_float_error;
                            
                        

                        } // iter end 

                        

                        printf("End train model_global\n");
                        communications_per_epoch = (uint64_t) (communications_CPU_DPU_per_epoch + communications_DPU_CPU_per_epoch);
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
                                    if (results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_init > tmp_max_time_init) {
                                        tmp_max_time_init = results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_init;
                                    }
                                    if (results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_compute > tmp_max_time_compute) {
                                        tmp_max_time_compute = results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_compute;
                                    }
                                    if (results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_mram_to_wram > tmp_max_time_mram_to_wram) {
                                        tmp_max_time_mram_to_wram = results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_mram_to_wram;
                                    }
                                    if (results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_wram_to_mram > tmp_max_time_wram_to_mram) {
                                        tmp_max_time_wram_to_mram = results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].cycles_wram_to_mram;
                                    }
                                    
                                    tmp_bytes_mram_to_wram += (double) (((double) (results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_mram_to_wram))/((double) (1e9)));
                                    
                                    tmp_bytes_wram_to_mram += (double) (((double) (results_ga[rep*NR_DPUS*NR_TASKLETS + m*NR_TASKLETS + x].bytes_wram_to_mram))/((double) (1e9)));
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
                        free(W_dpu_fp);

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
    

    free(error_rate);
    free(reduction);
    free(sum_of_Y);
    free(cross_entropy_loss);
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
