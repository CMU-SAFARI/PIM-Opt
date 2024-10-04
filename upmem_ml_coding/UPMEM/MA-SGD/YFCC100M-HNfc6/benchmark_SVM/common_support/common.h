#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdbool.h> 

// Data type
#ifdef UINT32
#define U uint32_t // quantized data has that type
#define MUL_U 2 // Shift left to divide by sizeof(T)
#define DIV_U 2 // Shift right to divide by sizeof(T)
#define LU uint64_t
#define MUL_LU 3
#define DIV_LU 3
#define S int32_t // quantized model has that type
#define MUL_S 2
#define DIV_S 2
#define LS int64_t 
#define MUL_LS 3
#define DIV_LS 3
#endif

// Structures used by both the host and the dpu to communicate information 
typedef struct {
    uint32_t n_size;
    uint32_t n_size_pad;
    uint32_t n_size_pad_W_fp;
    uint32_t n_size_features_frac; //number of features for each tasklet, i.e. a fraction of features of this tasklet
    uint32_t n_size_features_frac_pad; //align properly

    uint32_t batch_size_frac_Y_pad;
    uint32_t communicate_Y_pad;
    uint32_t current_batch_id;

    uint32_t max_rows;
    uint32_t nr_batches;
    uint32_t size_of_batch;
    uint32_t learning_rate;

    uint32_t task_epochs;

    uint32_t b_size_frac;

    uint32_t b_size_frac_log;

    uint32_t reg_term;
    uint32_t learning_rate_plus_reg_term;

    uint32_t samples_loop_transfer;
    uint32_t transfer;
    uint32_t samples_per_transfer;

 

} dpu_arguments_t;

// Specific information for each DPU
typedef struct {
    uint32_t rows_per_dpu;
    uint32_t rows_per_dpu_pad;
    uint32_t prev_rows_dpu;
} dpu_info_t;


// model average weights W
typedef struct {
    LS overflow_test;
    LS underflow_test;
    S* W_local_models_buffer;
    LS* W_global_model_buffer;

    S* W_dpu_fp;
    LS* bias_local_models_buffer;
    LS* bias_global_model_buffer;

    uint64_t number_of_features_per_thread;
    uint64_t n_size;
    uint64_t thread_id;
} model_average_parallel_t;


// parallelize train_error, test_error
typedef struct {
    float* error_rate;
    uint64_t* reduction;
    uint64_t* sum_of_Y;
    U* X;
    U* Y;
    float* W_host_float;
    float* bias_global_model_buffer;
    uint64_t m_size;
    uint64_t n_size;
} compute_error_rate_worker_t;

// parallelize train_loss, test_loss
typedef struct {
    double* hinge_loss;
    U* X;
    U* Y;
    float* W_host_float;
    float* bias_global_model_buffer;
    uint64_t m_size;
    uint64_t n_size;
} compute_loss_worker_t;


// Transfer size between MRAM and WRAM
#ifdef BL
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#else
#define BLOCK_SIZE_LOG2 8 //maximum transfer size would be #define BLOCK_SIZE_LOG2 10
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#endif


#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)

// fixed point arithmetic 
#define SHIFT_AMOUNT 31 
#define CAST 0x80000000
#define CONSTRAINT 0x00010000

#define N_FEATURES 4096 // representing number of features per sample
#define N_FRAC_FEATURES (N_FEATURES / NR_TASKLETS) // number of features handled per sample by a tasklet
#define N_FRAC_FEATURES_OVERFLOW (N_FEATURES % NR_TASKLETS) // in case of overflow, tasklet with highest id handles remaining features


// ADD --> Legend for Benchmark // code from Prim Benchmark + dpu_checksum from UPMEM

typedef struct {
    uint64_t cycles_init;
    
    uint64_t cycles_compute;

    uint64_t cycles_mram_to_wram;
    uint64_t bytes_mram_to_wram;

    uint64_t cycles_wram_to_mram;
    uint64_t bytes_wram_to_mram;

} dpu_result_t;

typedef struct {
    uint64_t cycles_init;
    
    uint64_t cycles_compute;

    uint64_t cycles_mram_to_wram;
    double bytes_mram_to_wram;

    uint64_t cycles_wram_to_mram;
    double bytes_wram_to_mram;

} dpu_result_per_epoch_t;


typedef struct {
    double time_init;
    
    double time_compute;

    double time_mram_to_wram;
    uint64_t bytes_mram_to_wram;

    double time_wram_to_mram;
    uint64_t bytes_wram_to_mram;

} dpu_result_double_t;

typedef struct {
    double time_init;
    
    double time_compute;

    double time_mram_to_wram;
    double giga_bytes_mram_to_wram;

    double time_wram_to_mram;
    double giga_bytes_wram_to_mram;

} dpu_final_result_double_t;


#endif
