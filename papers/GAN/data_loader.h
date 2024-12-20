#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <zlib.h>
#include <sys/stat.h>

#define CIFAR_IMAGE_SIZE 32
#define CIFAR_CHANNELS 3
#define CIFAR_PIXELS (CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE * CIFAR_CHANNELS)
#define CIFAR_TRAIN_SIZE 50000
#define CIFAR_BATCH_SIZE 10000
#define DOWNLOAD_PATH "cifar10_download.tar.gz"
#define EXTRACT_PATH "cifar10_data"
#define CIFAR_LABEL_SIZE 1
#define CIFAR_RECORD_SIZE (CIFAR_LABEL_SIZE + CIFAR_PIXELS)

static int progress_callback(void *clientp, double dltotal, double dlnow, double ultotal, double ulnow) {
    if (dltotal > 0) {
        printf("\rDownloading: %.1f%%", (dlnow/dltotal) * 100);
        fflush(stdout);
    }
    return 0;
}

unsigned char* download_cifar10() {
    char first_batch_path[256];
    snprintf(first_batch_path, sizeof(first_batch_path), 
             "%s/cifar-10-batches-bin/data_batch_1.bin", EXTRACT_PATH);
    
    FILE *test_file = fopen(first_batch_path, "rb");
    if (!test_file) {
        printf("Downloading CIFAR-10 dataset...\n");
        
        CURL *curl_handle;
        CURLcode res;
        
        FILE *fp = fopen(DOWNLOAD_PATH, "wb");
        if (!fp) {
            fprintf(stderr, "Failed to create download file\n");
            return NULL;
        }
        
        curl_global_init(CURL_GLOBAL_ALL);
        curl_handle = curl_easy_init();
        
        curl_easy_setopt(curl_handle, CURLOPT_URL, 
            "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz");
        curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, fwrite);
        curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, fp);
        curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl_handle, CURLOPT_PROGRESSFUNCTION, progress_callback);
        
        res = curl_easy_perform(curl_handle);
        fclose(fp);
        
        if(res != CURLE_OK) {
            fprintf(stderr, "Download failed: %s\n", curl_easy_strerror(res));
            curl_easy_cleanup(curl_handle);
            curl_global_cleanup();
            return NULL;
        }
        
        #ifdef _WIN32
            _mkdir(EXTRACT_PATH);
        #else
            mkdir(EXTRACT_PATH, 0777);
        #endif
        
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "tar xzf %s -C %s", DOWNLOAD_PATH, EXTRACT_PATH);
        system(cmd);
        
        curl_easy_cleanup(curl_handle);
        curl_global_cleanup();
    } else {
        fclose(test_file);
        printf("CIFAR-10 dataset already exists locally, skipping download...\n");
    }
    
    unsigned char* data = (unsigned char*)aligned_alloc(32, CIFAR_TRAIN_SIZE * CIFAR_PIXELS);
    if (!data) {
        fprintf(stderr, "Failed to allocate memory for dataset\n");
        return NULL;
    }
    memset(data, 0, CIFAR_TRAIN_SIZE * CIFAR_PIXELS);
    
    printf("Loading dataset from files...\n");
    unsigned char* temp_buffer = (unsigned char*)malloc(CIFAR_RECORD_SIZE);
    if (!temp_buffer) {
        fprintf(stderr, "Failed to allocate temporary buffer\n");
        free(data);
        return NULL;
    }

    for (int batch = 1; batch <= 5; batch++) {
        char batch_path[256];
        snprintf(batch_path, sizeof(batch_path), 
                 "%s/cifar-10-batches-bin/data_batch_%d.bin", EXTRACT_PATH, batch);
        
        FILE *batch_file = fopen(batch_path, "rb");
        if (!batch_file) {
            fprintf(stderr, "Failed to open batch file %d\n", batch);
            free(data);
            free(temp_buffer);
            return NULL;
        }
        
        for (size_t i = 0; i < CIFAR_BATCH_SIZE; i++) {
            size_t read_size = fread(temp_buffer, 1, CIFAR_RECORD_SIZE, batch_file);
            if (read_size != CIFAR_RECORD_SIZE) {
                fprintf(stderr, "Failed to read complete record %zu in batch %d\n", i, batch);
                free(data);
                free(temp_buffer);
                fclose(batch_file);
                return NULL;
            }
            
            size_t dest_offset = ((batch - 1) * CIFAR_BATCH_SIZE + i) * CIFAR_PIXELS;
            memcpy(data + dest_offset, temp_buffer + CIFAR_LABEL_SIZE, CIFAR_PIXELS);
        }
        
        fclose(batch_file);
        printf("\rLoaded batch %d/5", batch);
        fflush(stdout);
    }
    
    free(temp_buffer);
    printf("\nFinished loading dataset!\n");
    
    return data;
}

void load_batch(unsigned char* dataset, void* batch, size_t batch_size, size_t offset) {
    if (!dataset || !batch || offset >= CIFAR_TRAIN_SIZE || 
        batch_size > CIFAR_TRAIN_SIZE || offset + batch_size > CIFAR_TRAIN_SIZE) {
        fprintf(stderr, "Invalid parameters in load_batch\n");
        return;
    }
    
    float* float_batch = (float*)batch;
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < CIFAR_PIXELS; j++) {
            float_batch[i * CIFAR_PIXELS + j] = 
                (dataset[(offset + i) * CIFAR_PIXELS + j] / 127.5f) - 1.0f;
        }
    }
} 