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
    // if thread_id == 0 do set W_host_float[0] to zero
    // if thread_id == 62, set from W_host_float[1000001] to zero for thread_id == 62
    model_average_parallel_t* thread_info_list_current = (model_average_parallel_t*) thread_info_list;
    LS overflow_test = thread_info_list_current->overflow_test;
    LS underflow_test = thread_info_list_current->underflow_test;
    S* W_local_models_buffer = thread_info_list_current->W_local_models_buffer;
    LS* W_global_model_buffer = thread_info_list_current->W_global_model_buffer;

    S* W_host_float = thread_info_list_current->W_host_float;
    uint64_t number_of_features_per_thread = thread_info_list_current->number_of_features_per_thread;
    uint64_t n_size = thread_info_list_current->n_size;
    uint64_t thread_id = thread_info_list_current->thread_id;
    uint64_t bound_start_set_zero = thread_info_list_current->bound_start_set_zero;

    for (uint64_t m = 0; m < NR_DPUS; ++m) {
        for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
            W_global_model_buffer[x] += (LS) W_local_models_buffer[m*n_size + x];
        }
    }
    if (thread_id != 0 && thread_id != 62) {
        for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
            LS tmp = W_global_model_buffer[x] / NR_DPUS;
            if (tmp > overflow_test || tmp < underflow_test) {
                if (tmp > overflow_test) {
                    W_host_float[x] = (S) (overflow_test - 1);
                } else {
                    W_host_float[x] = (S) (underflow_test + 1);
                }
            } else {
                W_host_float[x] = (S) (tmp);
            }
        }
    } else if (thread_id == 0) {
        for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
            if (x != 0) {
                LS tmp = W_global_model_buffer[x] / NR_DPUS;
                if (tmp > overflow_test || tmp < underflow_test) {
                    if (tmp > overflow_test) {
                        W_host_float[x] = (S) (overflow_test - 1);
                    } else {
                        W_host_float[x] = (S) (underflow_test + 1);
                    }
                } else {
                    W_host_float[x] = (S) (tmp);
                }
            } else {
                W_host_float[x] = (S) (0);
            }
        }
    } else {
        for (uint64_t x = 0; x < number_of_features_per_thread; ++x) {
            if (x <= bound_start_set_zero) {
                LS tmp = W_global_model_buffer[x] / NR_DPUS;
                if (tmp > overflow_test || tmp < underflow_test) {
                    if (tmp > overflow_test) {
                        W_host_float[x] = (S) (overflow_test - 1);
                    } else {
                        W_host_float[x] = (S) (underflow_test + 1);
                    }
                } else {
                    W_host_float[x] = (S) (tmp);
                }
            } else {
                W_host_float[x] = (S) (0);
            }
        }
    }
    

    pthread_exit(NULL);
}




void* compute_cross_entropy_loss_parallel(void* thread_info_list) {
    compute_loss_worker_t* thread_info_list_current = (compute_loss_worker_t*) thread_info_list;
    double* cross_entropy_loss = thread_info_list_current->cross_entropy_loss;
    U* X = thread_info_list_current->X;
    U* Y = thread_info_list_current->Y;
    float* W_host_float = thread_info_list_current->W_host_float;
    uint64_t m_size = thread_info_list_current->m_size;
    uint64_t n_size = thread_info_list_current->n_size;
    uint64_t n_size_samples = thread_info_list_current->n_size_samples;

    for (uint64_t m = 0; m < m_size; ++m) {
        float dot_product = 0.0;
        uint64_t tmp = (uint64_t) (0);
        for (uint64_t n = 0; n < 40; ++n) { 
            if (n < 39) {
                tmp = (uint64_t) (X[m*n_size_samples + n] >> 2);
                dot_product += (W_host_float[tmp]);
            } else { // add bias
                dot_product += W_host_float[1000000]; // bias stored in 100000
            }
            
        }
        double sigmoid_tmp = (double) (1.0 / (1.0 + exp((double)(-dot_product)))); 
        
        double cross_entropy_loss_tmp = Y[m] == 1 ? log(sigmoid_tmp) : log(((double) (1.0)) - sigmoid_tmp);


        
        if (Y[m] == 0) {
            cross_entropy_loss_tmp *= (double) (0.5); 
        } else {
            cross_entropy_loss_tmp *= (double) (15.0); 
        }
        
        
        *cross_entropy_loss += cross_entropy_loss_tmp;
    }

    pthread_exit(NULL);
}



// Main of the Host Application
int main(int argc, char **argv) {
    // Initialize random seed
    srand(time(NULL));

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


    uint32_t array_b_size_frac[] = {1024, 2048, 4096, 8192};
    uint32_t array_b_size_frac_length = sizeof(array_b_size_frac) / sizeof(array_b_size_frac[0]);

    if (AE == 1) { // reproducing the results in our paper
        if (strong_scaling == 0) {
            if (NR_DPUS == 256 || NR_DPUS == 512 || NR_DPUS == 1024) {
                array_b_size_frac[0] = 2048;
                array_b_size_frac_length = 1;
            }
        }
        if (strong_scaling == 1) {
            array_b_size_frac[0] = 2048;
            array_b_size_frac_length = 1;
        }
    }


    
    uint32_t array_learning_rate[] = {0}; 
    uint32_t array_learning_rate_length = sizeof(array_learning_rate) / sizeof(array_learning_rate[0]);
    
    uint32_t array_reg_term[] = {12};
    uint32_t array_reg_term_length = sizeof(array_reg_term) / sizeof(array_reg_term[0]);

    uint32_t array_epochs[] = {10};
    uint32_t array_epochs_length = sizeof(array_epochs)/sizeof(array_epochs[0]);
    uint64_t max_epochs_length = (uint64_t) array_epochs[array_epochs_length-1];
    uint32_t array_task_epochs[] = {1};
    uint32_t array_task_epochs_length = sizeof(array_task_epochs) / sizeof(array_task_epochs[0]);

   


    uint64_t m_size = (uint64_t) (0);
    uint64_t number_of_test_samples = (uint64_t) (0);
    if (NR_DPUS == 256 || strong_scaling == 1) {
        m_size = (uint64_t) (50331648);
        number_of_test_samples = (uint64_t) (178236537);
    }
    else if (NR_DPUS == 512) {
        m_size = (uint64_t) (100663296);
        number_of_test_samples = (uint64_t) (178236537);
    }
    else if (NR_DPUS == 1024) {
        m_size = (uint64_t) (201326592);
        number_of_test_samples = (uint64_t) (178236537);
    }
    else if (NR_DPUS == 2048) {
        m_size = (uint64_t) (402653184);
        number_of_test_samples = (uint64_t) (178236537);
    }

    uint64_t n_size = (uint64_t) (N_FEATURES);

    uint64_t n_size_pad = (uint64_t) ((n_size*sizeof(U)) % 8) == 0 ? n_size : roundup(n_size, (8/sizeof(U)));
    uint64_t n_size_pad_W_fp = (uint64_t) ((n_size*sizeof(S)) % 8) == 0 ? n_size : roundup(n_size, (8/sizeof(S)));
    
    uint64_t n_size_samples = (uint64_t) (40);

    // Input/output allocation
    X = calloc((uint64_t) (m_size * n_size_samples), sizeof(U)); 
    Y = calloc(m_size, sizeof(U)); 

    
     
    X_test = (U*) calloc((uint64_t) (number_of_test_samples * n_size_samples), sizeof(U));
    Y_test = (U*) calloc(number_of_test_samples, sizeof(U));
    
    // init trainging dataset and weight for host 
    U *bufferX = X;
    U *bufferY = Y;

    U *bufferX_test = X_test;
    U *bufferY_test = Y_test;

    
    printf("Before reading test input\n");
    read_test_input_criteo(bufferX_test, bufferY_test, number_of_test_samples, strong_scaling);
    printf("\nBefore reading train input\n");
    read_input_criteo(bufferX, bufferY, m_size, strong_scaling);

    
    printf("Before starting the experiments\n");
    char* filename_experiment_dpu = (char*) calloc(1000, sizeof(char));
    float* W_checkpoint_for_compute_error = (float*) calloc(max_epochs_length*n_size, sizeof(float));
    
    double* compute_loss_worker_list = (double*) calloc(64, sizeof(double));
    float* W_dpu_float_error = (float*) calloc(n_size, sizeof(float));
    S* W_local_models_buffer = (S*) calloc(((uint64_t) (NR_DPUS))*n_size, sizeof(S));

    LS* W_global_model_buffer = (LS*) calloc(n_size, sizeof(LS));
    compute_loss_worker_t* thread_info_compute_loss_list = (compute_loss_worker_t*) calloc(64, sizeof(compute_loss_worker_t));

    // set compute_error_rate_worker_list to zero again after usage
    // set compute_loss_worker_list to zero again after usage
    uint64_t start_index_worker_train = ((uint64_t) m_size) / ((uint64_t) 64);
    uint64_t start_index_worker_test = ((uint64_t) number_of_test_samples) / ((uint64_t) 64);
    for (uint64_t s = 0; s < 64; ++s) {
        

        thread_info_compute_loss_list[s].cross_entropy_loss = compute_loss_worker_list + s;
        thread_info_compute_loss_list[s].m_size = start_index_worker_train;
        thread_info_compute_loss_list[s].n_size = (uint64_t) n_size;
        thread_info_compute_loss_list[s].n_size_samples = (uint64_t) n_size_samples;
    }

    
    
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
                        uint32_t nr_batches = (uint32_t) m_size/(b_size_frac*NR_DPUS);
                        
                        memset(filename_experiment_dpu, 0, 1000*sizeof(char));
                        memset(W_checkpoint_for_compute_error, 0, max_epochs_length*n_size*sizeof(float));
                        memset(W_dpu_float_error, 0, n_size*sizeof(float));
                        memset(W_local_models_buffer, 0, ((uint64_t) (NR_DPUS))*n_size*sizeof(S));
                        memset(W_global_model_buffer, 0, n_size*sizeof(LS));
                        memset(thread_info_compute_loss_list, 0, 64*sizeof(compute_loss_worker_t));
                        memset(cross_entropy_loss, 0, sizeof(double));


                        printf("About to create fdpu file\n");
                        if (strong_scaling == 0) {
                            sprintf(filename_experiment_dpu, "%s/results_UPMEM/Criteo/weak_scaling/mbsgd_LR_uint32/lr_reg_DPU_arch__criteo__uint32__MB_SGD_per_DPU__NR_DPUS_%u__NR_TASKLETS_%u__m_size_%lu__m_test_size_%lu__n_size_%lu__b_size_frac_%u__nr_batches_%u__learning_rate_%u__reg_term_%u__epochs_%u__task_epochs_%u.txt", DEST_DIR, NR_DPUS, NR_TASKLETS, m_size, number_of_test_samples, n_size, b_size_frac, nr_batches, learning_rate, reg_term, epochs, task_epochs);
                        } else {
                            sprintf(filename_experiment_dpu, "%s/results_UPMEM/Criteo/strong_scaling/mbsgd_LR_uint32/lr_reg_DPU_arch__criteo__uint32__MB_SGD_per_DPU__NR_DPUS_%u__NR_TASKLETS_%u__m_size_%lu__m_test_size_%lu__n_size_%lu__b_size_frac_%u__nr_batches_%u__learning_rate_%u__reg_term_%u__epochs_%u__task_epochs_%u.txt", DEST_DIR, NR_DPUS, NR_TASKLETS, m_size, number_of_test_samples, n_size, b_size_frac, nr_batches, learning_rate, reg_term, epochs, task_epochs);
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
                        
                        uint32_t batch_size_frac_Y_pad = (b_size_frac*sizeof(U) % 8) == 0 ? b_size_frac : roundup(b_size_frac, (8/sizeof(U)));

                        uint32_t size_of_batch = b_size_frac * nr_of_dpus;
                        

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
                            input_args[i].n_size_samples = (uint32_t) (n_size_samples);
                            input_args[i].batch_size_frac_Y_pad = batch_size_frac_Y_pad;


                            input_args[i].nr_batches = nr_batches;
                            input_args[i].learning_rate = learning_rate; 
                            input_args[i].reg_term = reg_term;
                            input_args[i].task_epochs = task_epochs;
                            input_args[i].b_size_frac = b_size_frac;
                            input_args[i].b_size_frac_log = (U) abs((int32_t)log2(b_size_frac));

                            input_args[i].global_epoch_current = (uint32_t) (0);
                            input_args[i].learning_rate_plus_reg_term = (uint32_t) (learning_rate+reg_term);

                            input_args[i].current_batch_id = (uint32_t) (0);
                            
                            
                        }

                        // init Weight for DPU 
                        S* W_dpu_fp = calloc(n_size_pad_W_fp, sizeof(S)); 
                        


                        LS overflow_test = (LS) CAST;
                        LS underflow_test = -overflow_test;

                        // model average with 32 threads --> that is the number of cores on our system
                        model_average_parallel_t* thread_info_model_average_parallel_list = (model_average_parallel_t*) calloc(63, sizeof(model_average_parallel_t));
                        uint64_t start_index_thread_model_average = (uint64_t) (15896);//(15648); // 1001472/64 = 15648
                        for (uint64_t s = 0; s < 63; ++s) {
                            thread_info_model_average_parallel_list[s].overflow_test = overflow_test;
                            thread_info_model_average_parallel_list[s].underflow_test = underflow_test;
                            thread_info_model_average_parallel_list[s].W_local_models_buffer = W_local_models_buffer + s * start_index_thread_model_average;
                            thread_info_model_average_parallel_list[s].W_global_model_buffer = W_global_model_buffer + s * start_index_thread_model_average;
                            thread_info_model_average_parallel_list[s].W_host_float = W_dpu_fp + s * start_index_thread_model_average;
                            thread_info_model_average_parallel_list[s].number_of_features_per_thread = (uint64_t) (15896); // number of features of model
                            thread_info_model_average_parallel_list[s].n_size = n_size_pad_W_fp;
                            thread_info_model_average_parallel_list[s].thread_id = s;
                            thread_info_model_average_parallel_list[s].bound_start_set_zero = (uint64_t) (0);
                            if (s == 62) {
                                thread_info_model_average_parallel_list[s].number_of_features_per_thread = (uint64_t) (15920);
                                thread_info_model_average_parallel_list[s].bound_start_set_zero = (uint64_t) (14448);
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
                            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferX + (((uint64_t) dpu_info[i].prev_rows_dpu) * n_size_samples))); 
                        }
                        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, \
                            max_rows_per_dpu * n_size_samples * sizeof(U), DPU_XFER_DEFAULT)); 
                        
                        i = 0;
                        DPU_FOREACH(dpu_set, dpu, i) {
                            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferY + dpu_info[i].prev_rows_dpu)); 
                        }
                        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                            max_rows_per_dpu * n_size_samples * sizeof(U), communicate_Y_pad * sizeof(U), DPU_XFER_DEFAULT));
                        
                        clock_gettime(CLOCK_MONOTONIC, &end);
                        elapsed_time = (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                        fprintf(fp_dpu, "Load input data to DPUs. Elapsed time is %.9f s\n", elapsed_time);
                        
                        clock_gettime(CLOCK_MONOTONIC, &start);
                        broadcast_tables(dpu_set);
                        clock_gettime(CLOCK_MONOTONIC, &end);
                        elapsed_time = (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                        fprintf(fp_dpu, "TransPimLib: Create and broadcast tables to DPUs. Elapsed time is %.9f s\n", elapsed_time);

                        
                        // ITERATION AT DPU
                        fprintf(fp_dpu, "Run program on DPUs\n"); 
                        for(uint64_t rep = 0; rep < epochs; ++rep) { 
                            printf("Global_epoch: %lu\n", rep);
                            double elapsed_time_loading_to_DPUs = (double) (0);
                            double elapsed_time_DPU_kernel_time = (double) (0);
                            double elapsed_time_retrieving_results = (double) (0);
                            double elapsed_time_model_averaging = (double) (0);
                            for (uint32_t current_batch_id = 0; current_batch_id < nr_batches; ++current_batch_id) { 

                                // Copy W 
                                clock_gettime(CLOCK_MONOTONIC, &start);
                                i = 0; 
                                DPU_FOREACH(dpu_set, dpu, i) {
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, W_dpu_fp)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                    max_rows_per_dpu * n_size_samples * sizeof(U) + communicate_Y_pad * sizeof(U), \
                                    n_size_pad_W_fp * sizeof(S), DPU_XFER_DEFAULT)); 

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
                                    DPU_ASSERT(dpu_prepare_xfer(dpu, W_local_models_buffer + ((uint64_t) (i)) * n_size_pad_W_fp)); 
                                }
                                DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
                                    max_rows_per_dpu * n_size_samples * sizeof(U) + communicate_Y_pad * sizeof(U), \
                                    n_size_pad_W_fp * sizeof(S), \
                                    DPU_XFER_DEFAULT));  
                                
                                clock_gettime(CLOCK_MONOTONIC, &end);
                                elapsed_time_retrieving_results += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                                
                                
                                

                                // Compute gradient
                                clock_gettime(CLOCK_MONOTONIC, &start);
                                memset(W_global_model_buffer, 0, n_size*sizeof(LS));


                                
                                // start_index for each thread 32 of them 
                                pthread_t model_average_worker_thread[63];
                                for (uint32_t s = 0; s < 63; ++s) {
                                    int32_t thread_return_value = pthread_create(&model_average_worker_thread[s], NULL, model_average_function, (void *) &thread_info_model_average_parallel_list[s]);
                                    if (thread_return_value) {
                                        printf("Failed to create thread, %d\n", thread_return_value);
                                    }
                                }
                                for (uint32_t s = 0; s < 63; ++s) {
                                    pthread_join(model_average_worker_thread[s], NULL);
                                }

                                clock_gettime(CLOCK_MONOTONIC, &end);
                                elapsed_time_model_averaging += (double) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
                            }
                            fprintf(fp_dpu, "Epoch %lu. Loading W_dpu_fp to DPUs. Elapsed time is %.9f s\n", rep, elapsed_time_loading_to_DPUs);
                            fprintf(fp_dpu, "Epoch %lu. DPU kernel time. Elapsed time is %.9f s\n", rep, elapsed_time_DPU_kernel_time);
                            fprintf(fp_dpu, "Epoch %lu. Retrieve the models of all the DPUs. Elapsed time is %.9f s\n", rep, elapsed_time_retrieving_results);
                            fprintf(fp_dpu, "Epoch %lu. Model averaging on CPU. Elapsed time is %.9f s\n", rep, elapsed_time_model_averaging);



                            
                            fprintf(fp_dpu, "Epoch %lu. Averaged model W_dpu_fp.\n", rep);
                            printf("Epoch %lu. Averaged model W_dpu_fp.\n", rep);
                            
                            for (uint64_t l = 0; l < n_size; ++l) {
                                W_dpu_float_error[l] = (float) W_dpu_fp[l] / CAST;
                                W_checkpoint_for_compute_error[rep*n_size + l] = W_dpu_float_error[l];
                            }
                            
                            for (uint32_t x = 0; x < 100; ++x) {
                                if (x == 0) {
                                    fprintf(fp_dpu, "First features:\n");
                                    printf("First features:\n");
                                }
                                if (x < 99) {
                                    fprintf(fp_dpu, "%.9f, ", W_dpu_float_error[x]);
                                    printf("%.9f, ", W_dpu_float_error[x]);
                                } else {
                                    fprintf(fp_dpu, "%.9f\n", W_dpu_float_error[x]);
                                    printf("%.9f\n", W_dpu_float_error[x]);
                                }
                            }
                            for (uint32_t x = 500636; x < 500836; ++x) {
                                if (x == 500636) {
                                    fprintf(fp_dpu, "Middle features:\n");
                                    printf("Middle features:\n");
                                }
                                if (x < 500835) {
                                    fprintf(fp_dpu, "%.9f, ", W_dpu_float_error[x]);
                                    printf("%.9f, ", W_dpu_float_error[x]);
                                } else {
                                    fprintf(fp_dpu, "%.9f\n", W_dpu_float_error[x]);
                                    printf("%.9f\n", W_dpu_float_error[x]);
                                }
                            }
                            for (uint32_t x = 999900; x < 1000000; ++x) {
                                if (x == 999900) {
                                    fprintf(fp_dpu, "Last features:\n");
                                    printf("Last features:\n");
                                }
                                if (x < 999999) {
                                    fprintf(fp_dpu, "%.9f, ", W_dpu_float_error[x]);
                                    printf("%.9f, ", W_dpu_float_error[x]);
                                } else {
                                    fprintf(fp_dpu, "%.9f\n", W_dpu_float_error[x]);
                                    printf("%.9f\n", W_dpu_float_error[x]);
                                }
                            }
                            fprintf(fp_dpu, "Bias = %.9f\n", W_dpu_float_error[1000000]);
                            printf("Bias = %.9f\n", W_dpu_float_error[1000000]);


                        } // iter end 

                        

                        printf("End train model_global\n");
                        fprintf(fp_dpu, "Gathering data on compute error and others\nThis is after the timing experiments have successfully been completed\n");

                        
                        for (uint64_t i = 0; i < epochs; ++i) { 
                            
                            for (uint64_t s = 0; s < 64; ++s) {
                                thread_info_compute_loss_list[s].W_host_float = W_checkpoint_for_compute_error+(i*n_size);
                            }



                            // Compute training cross entropy loss
                            fprintf(fp_dpu, "Epoch %lu. Compute training cross entropy loss.\n", i);
                            printf("Epoch %lu. Compute training cross entropy loss.\n", i);

                            memset(cross_entropy_loss, 0, sizeof(double));
                            memset(compute_loss_worker_list, 0, 64*sizeof(double));

                            // calculate loss of regularization term
                            double l2_loss = (double) (0);
                            for (uint64_t s = 0; s < n_size; ++s) {
                                l2_loss += (double) (W_checkpoint_for_compute_error[(i*n_size) + s] * W_checkpoint_for_compute_error[(i*n_size) + s]);
                            }

                            double reg_term_double = (double) pow(2, reg_term+1); 
                            l2_loss = l2_loss / reg_term_double;

                            for (uint64_t s = 0; s < 64; ++s) {
                                thread_info_compute_loss_list[s].cross_entropy_loss = compute_loss_worker_list + s;
                                thread_info_compute_loss_list[s].n_size = (uint64_t) n_size;
                                thread_info_compute_loss_list[s].n_size_samples = (uint64_t) n_size_samples;

                                thread_info_compute_loss_list[s].X = bufferX + s * start_index_worker_train * n_size_samples;
                                thread_info_compute_loss_list[s].Y = bufferY + s * start_index_worker_train;
                                thread_info_compute_loss_list[s].m_size = start_index_worker_train;
                                if ((m_size != (64*start_index_worker_train)) && (s == 63)) {
                                    thread_info_compute_loss_list[s].m_size = m_size - (63 * start_index_worker_train);
                                }
                            }
                            pthread_t compute_loss_worker_thread[64];
                            for (uint32_t s = 0; s < 64; ++s) {
                                int32_t thread_return_value = pthread_create(&compute_loss_worker_thread[s], NULL, compute_cross_entropy_loss_parallel, (void *) &thread_info_compute_loss_list[s]);
                                if (thread_return_value) {
                                    printf("Failed to create thread, %d\n", thread_return_value);
                                }
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                pthread_join(compute_loss_worker_thread[s], NULL);
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                *cross_entropy_loss += compute_loss_worker_list[s];
                            }
                            *cross_entropy_loss = *cross_entropy_loss / ((uint64_t) m_size);
                            *cross_entropy_loss = -(*cross_entropy_loss);
                            double only_cross_entropy_loss = *cross_entropy_loss;
                            *cross_entropy_loss += l2_loss;
                            fprintf(fp_dpu, "Epoch %lu. Training cross entropy loss of averaged model = %.9f, only cross entropy loss = %.9f, only l2 loss = %.9f\n", i, *cross_entropy_loss, only_cross_entropy_loss, l2_loss);
                            printf("Epoch %lu. Training cross entropy loss of averaged model = %.9f, only cross entropy loss = %.9f, only l2 loss = %.9f\n", i, *cross_entropy_loss, only_cross_entropy_loss, l2_loss);
                            // Compute test cross entropy loss
                            memset(cross_entropy_loss, 0, sizeof(double));
                            memset(compute_loss_worker_list, 0, 64*sizeof(double));
                            fprintf(fp_dpu, "Epoch %lu. Compute test cross entropy loss.\n", i);
                            printf("Epoch %lu. Compute test cross entropy loss.\n", i);
                            for (uint64_t s = 0; s < 64; ++s) {
                                thread_info_compute_loss_list[s].cross_entropy_loss = compute_loss_worker_list + s;
                                thread_info_compute_loss_list[s].n_size = (uint64_t) n_size;
                                thread_info_compute_loss_list[s].n_size_samples = (uint64_t) n_size_samples;


                                thread_info_compute_loss_list[s].X = bufferX_test + s * start_index_worker_test * n_size_samples;
                                thread_info_compute_loss_list[s].Y = bufferY_test + s * start_index_worker_test;
                                thread_info_compute_loss_list[s].m_size = start_index_worker_test;
                                if ((m_size != (64*start_index_worker_test)) && (s == 63)) {
                                    thread_info_compute_loss_list[s].m_size = number_of_test_samples - (63 * start_index_worker_test);
                                }
                            }
                            pthread_t compute_loss_test_worker_thread[64];
                            for (uint32_t s = 0; s < 64; ++s) {
                                int32_t thread_return_value = pthread_create(&compute_loss_test_worker_thread[s], NULL, compute_cross_entropy_loss_parallel, (void *) &thread_info_compute_loss_list[s]);
                                if (thread_return_value) {
                                    printf("Failed to create thread, %d\n", thread_return_value);
                                }
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                pthread_join(compute_loss_test_worker_thread[s], NULL);
                            }
                            for (uint32_t s = 0; s < 64; ++s) {
                                *cross_entropy_loss += compute_loss_worker_list[s];
                            }
                            *cross_entropy_loss = *cross_entropy_loss / ((uint64_t) number_of_test_samples);
                            *cross_entropy_loss = -(*cross_entropy_loss);
                            only_cross_entropy_loss = *cross_entropy_loss;
                            *cross_entropy_loss += l2_loss;
                            fprintf(fp_dpu, "Epoch %lu. Test cross entropy loss of averaged model = %.9f, only cross entropy loss = %.9f, only l2 loss = %.9f\n\n", i, *cross_entropy_loss, only_cross_entropy_loss, l2_loss);
                            printf("Epoch %lu. Test cross entropy loss of averaged model = %.9f, only cross entropy loss = %.9f, only l2 loss = %.9f\n\n", i, *cross_entropy_loss, only_cross_entropy_loss, l2_loss);
                        }

                        fprintf(fp_dpu, "\n\n\n\n\n\n\n\n\n\n");
                        fprintf(fp_dpu, "Printing complete models for each global epoch with bias at location 0\n");
                        for (uint64_t i = 0; i < epochs; ++i) { 
                            uint64_t start_print_index = (uint64_t) (i*n_size);
                            fprintf(fp_dpu, "Epoch %lu. Model W_dpu_fp.\n", i);
                            for (uint64_t j = 0; j < 1000000; ++j) {
                                if (j == 0) {
                                    fprintf(fp_dpu,"%.9f,", W_checkpoint_for_compute_error[start_print_index + 1000000]);
                                } else if (j == 999999) {
                                    fprintf(fp_dpu,"%.9f\n", W_checkpoint_for_compute_error[start_print_index + j]);
                                } else {
                                    fprintf(fp_dpu,"%.9f,", W_checkpoint_for_compute_error[start_print_index + j]);
                                }
                            }
                        }
                        

                        

                        // Deallocation
                        free(input_args); 
                        free(dpu_info);
                        free(W_dpu_fp);  
                        
                        fprintf(fp_dpu, "\n\n\n\n\n");
                        fprintf(fp_dpu, "Experiment successfully completed.\n");
                        fclose(fp_dpu);

                        printf("Done with: Criteo__uint32_MiniBatch_SGD_per_DPU__NR_DPUS_%u__m_size_%lu__n_size_%lu__b_size_frac_%u__nr_batches_%u__learning_rate_%u__epochs_%u__task_epochs_%u\n", NR_DPUS, m_size, n_size, b_size_frac, nr_batches, learning_rate, epochs, task_epochs);

                        
                    }
                }
            }
        }
    }
    


    free(cross_entropy_loss);
    free(W_local_models_buffer);
    free(W_global_model_buffer);
    free(thread_info_compute_loss_list);
    free(W_checkpoint_for_compute_error);
    free(compute_loss_worker_list);
    free(W_dpu_float_error);
    free(filename_experiment_dpu);
    free(X);
    free(Y);


    DPU_ASSERT(dpu_free(dpu_set));
    return 0;

    
}
