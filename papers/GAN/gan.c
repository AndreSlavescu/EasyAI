/*
GAN: Generative Adverserial Nets

GAN paper: https://arxiv.org/pdf/1406.2661
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "math_utils.h"
#include "data_loader.h"
#include <omp.h>

// png writer
// taken from https://github.com/nothings/stb/blob/master/stb_image_write.h
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct MLP {
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    float* weights;
    float* biases;
};

struct Generator {
    struct MLP network;
    size_t latent_dim;
    size_t output_dim;
};

struct Discriminator {
    struct MLP network;
    size_t input_dim;
};

void sample_noise(void* z, size_t size) {
    float* noise = (float*)z;
    for (size_t i = 0; i < size; i++) {
        noise[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

void initialize_weights(float* weights, size_t size) {
    const float scale = 0.1f;
    for (size_t i = 0; i < size; i++) {
        weights[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
    }
}

struct MLP create_mlp(size_t input_size, size_t hidden_size, size_t output_size) {
    struct MLP mlp = {
        .input_size = input_size,
        .hidden_size = hidden_size,
        .output_size = output_size,
        .weights = malloc(sizeof(float) * (input_size * hidden_size + hidden_size * output_size)),
        .biases = malloc(sizeof(float) * (hidden_size + output_size))
    };
    
    size_t total_weights = (input_size * hidden_size + hidden_size * output_size);
    size_t total_biases = (hidden_size + output_size);
    initialize_weights(mlp.weights, total_weights);
    initialize_weights(mlp.biases, total_biases);
    
    return mlp;
}

void generator_forward(struct Generator* g, void* input, void* output) {
    float* in = (float*)input;
    float* out = (float*)output;
    float* weights = g->network.weights;
    float* biases = g->network.biases;
    
    void* hidden = malloc(g->network.hidden_size * sizeof(float));
    void* layer_output = malloc(g->network.hidden_size * sizeof(float));
    float* h = (float*)hidden;
    
    #pragma omp parallel for
    for (size_t i = 0; i < g->network.hidden_size; i++) {
        h[i] = biases[i];
        for (size_t j = 0; j < g->network.input_size; j++) {
            h[i] += weights[i * g->network.input_size + j] * in[j];
        }
    }
    
    relu_act(hidden, hidden, g->network.hidden_size, sizeof(float));
    batchnorm2d(hidden, layer_output, g->network.hidden_size, sizeof(float));
    
    size_t weight_offset = g->network.input_size * g->network.hidden_size;
    #pragma omp parallel for
    for (size_t i = 0; i < g->network.output_size; i++) {
        out[i] = biases[g->network.hidden_size + i];
        for (size_t j = 0; j < g->network.hidden_size; j++) {
            out[i] += weights[weight_offset + i * g->network.hidden_size + j] * ((float*)layer_output)[j];
        }
    }
    tanh_act(output, output, g->output_dim, sizeof(float));
    
    free(hidden);
    free(layer_output);
}

void discriminator_forward(struct Discriminator* d, void* input, void* output) {
    float* in = (float*)input;
    float* out = (float*)output;
    float* weights = d->network.weights;
    float* biases = d->network.biases;
    
    void* hidden = malloc(d->network.hidden_size * sizeof(float));
    void* layer_output = malloc(d->network.hidden_size * sizeof(float));
    float* h = (float*)hidden;
    
    #pragma omp parallel for
    for (size_t i = 0; i < d->network.hidden_size; i++) {
        h[i] = biases[i];
        for (size_t j = 0; j < d->network.input_size; j++) {
            h[i] += weights[i * d->network.input_size + j] * in[j];
        }
    }
    
    relu_act(hidden, layer_output, d->network.hidden_size, sizeof(float));
    batchnorm2d(layer_output, hidden, d->network.hidden_size, sizeof(float));
    
    size_t weight_offset = d->network.input_size * d->network.hidden_size;
    #pragma omp parallel for
    for (size_t i = 0; i < d->network.output_size; i++) {
        out[i] = biases[d->network.hidden_size + i];
        for (size_t j = 0; j < d->network.hidden_size; j++) {
            out[i] += weights[weight_offset + i * d->network.hidden_size + j] * ((float*)hidden)[j];
        }
    }
    
    for (size_t i = 0; i < d->network.output_size; i++) {
        out[i] = 1.0f / (1.0f + expf(-out[i]));
    }
    
    free(hidden);
    free(layer_output);
}

double binary_cross_entropy(void* pred, void* target, size_t size) {
    float* p = (float*)pred;
    float* t = (float*)target;
    double loss = 0.0;
    const float epsilon = 1e-7f;
    for (size_t i = 0; i < size; i++) {
        float pred_val = fmaxf(fminf(p[i], 1.0f - epsilon), epsilon);
        loss += t[i] * log(pred_val) + (1 - t[i]) * log(1.0f - pred_val);
    }
    return -loss / size;
}

void compute_gradients(void* output, void* target, void* gradients, size_t size) {
    float* out = (float*)output;
    float* tgt = (float*)target;
    float* grad = (float*)gradients;
    
    const float epsilon = 1e-7f;
    for (size_t i = 0; i < size; i++) {
        float pred_val = fmaxf(fminf(out[i], 1.0f - epsilon), epsilon);
        grad[i] = (pred_val - tgt[i]) / (pred_val * (1.0f - pred_val) + epsilon);
    }
}

void update_weights(void* weights, void* gradients, size_t size, double learning_rate) {
    float* w = (float*)weights;
    float* g = (float*)gradients;
    
    const float max_grad = 1.0f;
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        float grad = fmaxf(fminf(g[i], max_grad), -max_grad);
        w[i] -= learning_rate * grad;
    }
}

void save_weights(const char* filename, struct MLP* mlp) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }
    
    size_t total_weights = (mlp->input_size * mlp->hidden_size + mlp->hidden_size * mlp->output_size);
    size_t total_biases = (mlp->hidden_size + mlp->output_size);
    
    fwrite(mlp->weights, sizeof(float), total_weights, f);
    fwrite(mlp->biases, sizeof(float), total_biases, f);
    fclose(f);
}

void load_weights(const char* filename, struct MLP* mlp) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open weights file: %s\n", filename);
        return;
    }
    
    size_t total_weights = (mlp->input_size * mlp->hidden_size + mlp->hidden_size * mlp->output_size);
    size_t total_biases = (mlp->hidden_size + mlp->output_size);
    
    fread(mlp->weights, sizeof(float), total_weights, f);
    fread(mlp->biases, sizeof(float), total_biases, f);
    fclose(f);
}

void generate_sample(struct Generator* g) {
    void* latent = malloc(g->latent_dim * sizeof(float));
    void* image = malloc(CIFAR_PIXELS * sizeof(float));
    
    sample_noise(latent, g->latent_dim);
    generator_forward(g, latent, image);
    
    unsigned char* img_bytes = malloc(CIFAR_PIXELS);
    float* img_float = (float*)image;
    for (int i = 0; i < CIFAR_PIXELS; i++) {
        img_bytes[i] = (unsigned char)((img_float[i] + 1.0f) * 127.5f);
    }
    
    stbi_write_png("sample.png", 32, 32, 3, img_bytes, 32 * 3);
    printf("Generated sample saved to sample.png\n");
    
    free(img_bytes);
    free(latent);
    free(image);
}

void print_progress_bar(int progress, int total, int epoch, double d_loss, double g_loss) {
    const int bar_width = 50;
    float percentage = (float)progress / total;
    int filled = (int)(bar_width * percentage);

    printf("\rEpoch %d [", epoch);
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else printf(" ");
    }
    printf("] %.1f%% - D_loss: %.4f, G_loss: %.4f", 
           percentage * 100.0f, d_loss, g_loss);
    fflush(stdout);
}

int main(int argc, char** argv) {
    bool training_mode = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0) {
            training_mode = true;
        }
    }

    struct Generator g = {
        .network = create_mlp(100, 512, CIFAR_PIXELS),
        .latent_dim = 100,
        .output_dim = CIFAR_PIXELS
    };
    
    struct Discriminator d = {
        .network = create_mlp(CIFAR_PIXELS, 512, 1),
        .input_dim = CIFAR_PIXELS
    };

    if (training_mode) {
        printf("Downloading the CIFAR-10 image dataset...\n");
        unsigned char* cifar_data = download_cifar10();
        if (!cifar_data) {
            fprintf(stderr, "Failed to download CIFAR-10\n");
            return 1;
        }
        printf("Finished downloading the CIFAR-10 image dataset!\n");
        
        const int num_epochs = 100;
        const int batch_size = 64;
        const double learning_rate = 0.0002;
        
        void* fake_images = malloc(batch_size * CIFAR_PIXELS * sizeof(float));
        void* real_images = malloc(batch_size * CIFAR_PIXELS * sizeof(float));
        void* latent_vectors = malloc(batch_size * g.latent_dim * sizeof(float));
        void* d_predictions = malloc(batch_size * sizeof(float));
        
        printf("Beginning training...\n");
        
        void* d_gradients = malloc(d.network.input_size * d.network.hidden_size * sizeof(float));
        void* g_gradients = malloc(g.network.input_size * g.network.hidden_size * sizeof(float));
        
        float* real_labels = malloc(batch_size * sizeof(float));
        float* fake_labels = malloc(batch_size * sizeof(float));
        for (int i = 0; i < batch_size; i++) {
            real_labels[i] = 1.0f;
            fake_labels[i] = 0.0f;
        }
        
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            double epoch_d_loss = 0.0;
            double epoch_g_loss = 0.0;
            
            for (int batch = 0; batch < CIFAR_TRAIN_SIZE / batch_size; batch++) {
                load_batch(cifar_data, real_images, batch_size, batch * batch_size);
                
                discriminator_forward(&d, real_images, d_predictions);
                compute_gradients(d_predictions, real_labels, d_gradients, batch_size);
                update_weights(d.network.weights, d_gradients, d.network.hidden_size, learning_rate);
                epoch_d_loss += binary_cross_entropy(d_predictions, real_labels, batch_size);
                
                generator_forward(&g, latent_vectors, fake_images);
                discriminator_forward(&d, fake_images, d_predictions);
                compute_gradients(d_predictions, fake_labels, d_gradients, batch_size);
                update_weights(d.network.weights, d_gradients, d.network.hidden_size, learning_rate);
                epoch_d_loss += binary_cross_entropy(d_predictions, fake_labels, batch_size);
                
                sample_noise(latent_vectors, batch_size * g.latent_dim);
                generator_forward(&g, latent_vectors, fake_images);
                discriminator_forward(&d, fake_images, d_predictions);
                compute_gradients(d_predictions, real_labels, g_gradients, batch_size);
                update_weights(g.network.weights, g_gradients, g.network.hidden_size, learning_rate);
                epoch_g_loss += binary_cross_entropy(d_predictions, real_labels, batch_size);
                
                if (batch % 10 == 0) {
                    double current_d_loss = epoch_d_loss / (2.0 * (batch + 1) * batch_size);
                    double current_g_loss = epoch_g_loss / ((batch + 1) * batch_size);
                    print_progress_bar(batch, CIFAR_TRAIN_SIZE / batch_size, 
                                     epoch, current_d_loss, current_g_loss);
                }
            }
            
            print_progress_bar(CIFAR_TRAIN_SIZE / batch_size, 
                             CIFAR_TRAIN_SIZE / batch_size,
                             epoch,
                             epoch_d_loss / (2.0 * CIFAR_TRAIN_SIZE / batch_size),
                             epoch_g_loss / (CIFAR_TRAIN_SIZE / batch_size));
            printf("\n");
        }
        
        free(fake_labels);
        free(real_labels);
        free(g_gradients);
        free(d_gradients);
        free(d_predictions);
        free(latent_vectors);
        free(real_images);
        free(fake_images);
        free(cifar_data);
        free(d.network.biases);
        free(d.network.weights);
        free(g.network.biases);
        free(g.network.weights);
        
        save_weights("generator_weights.bin", &g.network);
        save_weights("discriminator_weights.bin", &d.network);
    } else {
        load_weights("generator_weights.bin", &g.network);
        generate_sample(&g);
    }
    
    return 0;
}