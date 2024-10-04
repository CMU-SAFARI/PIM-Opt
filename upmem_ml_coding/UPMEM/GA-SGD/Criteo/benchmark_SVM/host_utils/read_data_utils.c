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


void read_input_criteo(U *X, U *Y, uint64_t length, uint64_t strong_scaling) {
    printf("Reading training dataset from yfcc_..._label_features.txt\n");


    uint64_t counter = (uint64_t) (0); 
    uint64_t counter_segmentation_faults = (uint64_t) (0);
    uint64_t n_size = (uint64_t) (40);

    FILE* fp;
    char row[MAXCHAR];
    char *token;

    char data_file_path[1000];
    if (NR_DPUS == 256 || strong_scaling == 1) {
        sprintf(data_file_path, "%s/Criteo_train_NR_DPUS_256_label_0_48610305_label_1_1721343_total_50331648.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 512) {
        sprintf(data_file_path, "%s/Criteo_train_NR_DPUS_512_label_0_97220610_label_1_3442686_total_100663296.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 1024) {
        sprintf(data_file_path, "%s/Criteo_train_NR_DPUS_1024_label_0_194441220_label_1_6885372_total_201326592.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 2048) {
        sprintf(data_file_path, "%s/Criteo_train_NR_DPUS_2048_label_0_388882440_label_1_13770744_total_402653184.txt", SOURCE_DIR);
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
        

        for (uint64_t i = 0; i < 40; ++i) {
            if (i < 39) {
                if (token != NULL) {
                    sscanf(token, "%u", &X[tmp_index +i]);
                } else {
                    ++ counter_segmentation_faults;
                    printf("Avoided segmentation fault at counter = %lu\n", counter);
                    continue;
                }
                X[tmp_index +i] = (U) ((X[tmp_index +i]) << 2);
                token = strtok(NULL, " ");

            } else {
                X[tmp_index + i] = (U) (0);
            }
            
        }

        
        counter++;
        
    }
    fclose(fp);
    printf("Number of segmentation faults = %lu\n", counter_segmentation_faults);
    printf("\nSuccessfully generate train data\n");
}


void read_test_input_criteo(U *X, U *Y, uint64_t length, uint64_t strong_scaling) {
    printf("Reading test dataset from yfcc_..._label_features.txt\n");

    uint64_t counter = (uint64_t) (0); 
    uint64_t counter_segmentation_faults = (uint64_t) (0);
    uint64_t n_size = (uint64_t) (40);

    FILE* fp;
    char row[MAXCHAR];
    char *token;

    char data_file_path[1000];
    if (NR_DPUS == 256 || strong_scaling == 1) {
        sprintf(data_file_path, "%s/test_data_criteo_tb_day_23.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 512) {
        sprintf(data_file_path, "%s/test_data_criteo_tb_day_23.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r"); 
    }
    else if (NR_DPUS == 1024) {
        sprintf(data_file_path, "%s/test_data_criteo_tb_day_23.txt", SOURCE_DIR);
        fp = fopen(data_file_path, "r");
    }
    else if (NR_DPUS == 2048) {
        sprintf(data_file_path, "%s/test_data_criteo_tb_day_23.txt", SOURCE_DIR);
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
        

        for (uint64_t i = 0; i < 40; ++i) {
            if (i < 39) {
                if (token != NULL) {
                    sscanf(token, "%u", &X[tmp_index +i]);
                } else {
                    ++counter_segmentation_faults;
                    printf("Avoided segmentation fault at counter = %lu\n", counter);
                    continue;
                }
                X[tmp_index +i] = (U) ((X[tmp_index +i]) << 2);
                token = strtok(NULL, " ");

            } else {
                X[tmp_index + i] = (U) (0);
            }
            
        }

        
        counter++;
        
    }
    fclose(fp);
    printf("Number of segmentation faults = %lu\n", counter_segmentation_faults);
    printf("\nSuccessfully generate test data\n");

}





