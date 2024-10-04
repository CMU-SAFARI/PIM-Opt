#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>


#include <unistd.h>
#include <getopt.h>
#include <assert.h>


#include <math.h>
#include <time.h>

#include "../common_support/common.h"

#include "../host_utils/read_data_utils.h"


void read_input_yfcc100m(U *X, U *Y, uint64_t length, uint64_t strong_scaling) {
    printf("Reading training dataset from yfcc_..._label_features.txt\n");


    uint64_t counter = (uint64_t) (0); 
    uint64_t counter_segmentation_faults = (uint64_t) (0);
    uint64_t n_size = (uint64_t) (N_FEATURES);

    FILE* fp;
    char row[MAXCHAR];
    char *token;

    char data_file_path[1000];
    if (NR_DPUS == 256 || strong_scaling == 1) {
        sprintf(data_file_path, "%s/YFCC100M_train_corresponding_to_NR_DPUs_256_label_0_425984_label_1_425984_total_851968_uint32.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 512) {
        sprintf(data_file_path, "%s/YFCC100M_train_corresponding_to_NR_DPUs_512_label_0_851968_label_1_851968_total_1703936_uint32.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 1024) {
        sprintf(data_file_path, "%s/YFCC100M_train_corresponding_to_NR_DPUs_1024_label_0_1703936_label_1_1703936_total_3407872_uint32.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 2048) {
        sprintf(data_file_path, "%s/YFCC100M_train_corresponding_to_NR_DPUs_2048_label_0_3407872_label_1_3407872_total_6815744_uint32.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    
    if (fp == NULL) {
        perror("Can't open file!");
    }

    while (fgets(row, MAXCHAR, fp)) {
        if (counter == length) {
            break;
        }
        token = strtok(row, " ");
        char temp = atoi(token);
        token = strtok(NULL, " ");
        Y[counter] = temp == 1 ? ((U) (1)) : ((U) (0));
        uint64_t tmp_index = (uint64_t) (counter*n_size);
        

        for (uint64_t i = 0; i < n_size; ++i) {
            if (token != NULL) {
                sscanf(token, "%u", &X[tmp_index +i]);
            } else {
                ++ counter_segmentation_faults;
                printf("Avoided segmentation fault at counter = %lu\n", counter);
                continue;
            }
            X[tmp_index +i] = (U) (X[tmp_index +i]);
            token = strtok(NULL, " ");

        }

        counter++;
        
    }
    fclose(fp);
    printf("Number of segmentation faults = %lu\n", counter_segmentation_faults);
    printf("\nSuccessfully generate train data\n");
}

void read_test_input_yfcc100m(U *X, U *Y, uint64_t length, uint64_t strong_scaling) {
    printf("Reading test dataset from yfcc_..._label_features.txt\n");

    uint64_t counter = (uint64_t) (0); 
    uint64_t counter_segmentation_faults = (uint64_t) (0);
    uint64_t n_size = (uint64_t) (N_FEATURES);

    FILE* fp;
    char row[MAXCHAR];
    char *token;

    char data_file_path[1000];
    if (NR_DPUS == 256 || strong_scaling == 1) {
        sprintf(data_file_path, "%s/YFCC100M_test_corresponding_to_NR_DPUs_256_label_0_106496_label_1_106496_total_212992_uint32.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 512) {
        sprintf(data_file_path, "%s/YFCC100M_test_corresponding_to_NR_DPUs_512_label_0_212992_label_1_212992_total_425984_uint32.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 1024) {
        sprintf(data_file_path, "%s/YFCC100M_test_corresponding_to_NR_DPUs_1024_label_0_425984_label_1_425984_total_851968_uint32.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 2048) {
        sprintf(data_file_path, "%s/YFCC100M_test_corresponding_to_NR_DPUs_2048_label_0_851968_label_1_851968_total_1703936_uint32.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    
    if (fp == NULL) {
        printf("File cannot be opened\n");
        perror("Can't open file!");
    }

    while (fgets(row, MAXCHAR, fp)) {
        if (counter == length) {
            break;
        }
        token = strtok(row, " ");
        char temp = atoi(token);
        token = strtok(NULL, " ");
        Y[counter] = temp == 1 ? ((U) (1)) : ((U) (0));
        uint64_t tmp_index = (uint64_t) (counter*n_size);
        

        for (uint64_t i = 0; i < n_size; ++i) {
            if (token != NULL) {
                sscanf(token, "%u", &X[tmp_index +i]);
            } else {
                ++ counter_segmentation_faults;
                printf("Avoided segmentation fault at counter = %lu\n", counter);
                continue;
            }
            X[tmp_index +i] = (U) (X[tmp_index +i]);
            token = strtok(NULL, " ");

        }

        counter++;
        
    }
    fclose(fp);
    printf("Number of segmentation faults = %lu\n", counter_segmentation_faults);
    printf("\nSuccessfully generate test data\n");

}




