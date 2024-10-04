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
    uint32_t n_size_samples;

    uint32_t batch_size_frac_Y_pad;
    uint32_t communicate_Y_pad;

    uint32_t max_rows;
    uint32_t nr_batches;
    uint32_t learning_rate;

    uint32_t task_epochs;

    uint32_t b_size_frac;

    uint32_t b_size_frac_log;

    uint32_t reg_term;

    uint32_t global_epoch_current; 
    uint32_t learning_rate_plus_reg_term;
    
    uint32_t current_batch_id;
    uint32_t dummy3;



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

    S* W_host_float;

    uint64_t number_of_features_per_thread;
    uint64_t n_size;
    uint64_t thread_id;
    uint64_t bound_start_set_zero; // we have 1001472 features, set all features to zero from 1000001, since they are not part of the model
} model_average_parallel_t;





// parallelize train_loss, test_loss
typedef struct {
    double* cross_entropy_loss;
    U* X;
    U* Y;
    float* W_host_float;
    uint64_t m_size;
    uint64_t n_size;
    uint64_t n_size_samples;
} compute_loss_worker_t;




#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)

// fixed point arithmetic 
#define SHIFT_AMOUNT 31 
#define CAST 0x80000000


#define N_FEATURES 1001472 // representing number of features per sample

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
