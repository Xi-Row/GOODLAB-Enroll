// A whole cuda code:
//aim at test the LayerNorm Kernel
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARP_SIZE 32
#define EPSILON 1e-6f

//Warp reduce
__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// LayerNorm CUDA Kernel
__global__ void layer_norm_kernel(float* __restrict__ output, 
                                  const float* __restrict__ input, 
                                  const float* __restrict__ gamma, 
                                  const float* __restrict__ beta, 
                                  int hidden_size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane = tid % WARP_SIZE;  
    int warp_id = tid / WARP_SIZE;  

    int num_warps = blockDim.x / WARP_SIZE;
    int num_vec4 = hidden_size / 4;  

    __shared__ float buffer[2][WARP_SIZE];

    float4* input4  = (float4*) (input + bid * hidden_size);
    float4* output4 = (float4*) (output + bid * hidden_size);
    float4* gamma4  = (float4*) gamma;
    float4* beta4   = (float4*) beta;

    float sum = 0.0f, sum_sq = 0.0f;
    float4 local_data;

    for (int i = tid; i < num_vec4; i += blockDim.x) {
        local_data = input4[i];
        float v1 = local_data.x, v2 = local_data.y;
        float v3 = local_data.z, v4 = local_data.w;

        sum += (v1 + v2 + v3 + v4);
        sum_sq += (v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4);
    }

    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    if (lane == 0) {
        buffer[0][warp_id] = sum;
        buffer[1][warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < num_warps) ? buffer[0][tid] : 0;
        sum_sq = (tid < num_warps) ? buffer[1][tid] : 0;

        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);

        if (tid == 0) {
            float mean = sum / hidden_size;
            float var = sum_sq / hidden_size - mean * mean;
            buffer[0][0] = mean;
            buffer[1][0] = rsqrtf(var + EPSILON);
        }
    }
    __syncthreads();

    float mean = buffer[0][0];
    float rstd = buffer[1][0];

    for (int i = tid; i < num_vec4; i += blockDim.x) {
        float4 g = gamma4[i];
        float4 b = beta4[i];

        local_data.x = (local_data.x - mean) * rstd * g.x + b.x;
        local_data.y = (local_data.y - mean) * rstd * g.y + b.y;
        local_data.z = (local_data.z - mean) * rstd * g.z + b.z;
        local_data.w = (local_data.w - mean) * rstd * g.w + b.w;

        output4[i] = local_data;
    }
}

// launch Kernel
void launch_layer_norm(float* output, const float* input, const float* gamma, const float* beta, int batch, int hidden_size) {
    int threads = 256;  
    int blocks = batch;
    layer_norm_kernel<<<blocks, threads>>>(output, input, gamma, beta, hidden_size);
    cudaDeviceSynchronize();
}
//mean and variance after calculating LayerNorm
void compute_mean_std(float* data, int hidden_size) {
    float sum = 0.0f, sum_sq = 0.0f;

    for (int i = 0; i < hidden_size; i++) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }

    float mean = sum / hidden_size;
    float var = sum_sq / hidden_size - mean * mean;
    float std = sqrtf(var);

    printf("Mean: %.6f, Std Dev: %.6f\n", mean, std);
}

//Test Kernel
void test_layer_norm() {
    int batch = 2;        //disposing 2 batch
    int hidden_size = 128; //must be Multiples of four

    size_t size = batch * hidden_size * sizeof(float);

    // allocate host memory
    float *h_input  = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    float *h_gamma  = (float*)malloc(hidden_size * sizeof(float));
    float *h_beta   = (float*)malloc(hidden_size * sizeof(float));

    // initialize data
    for (int i = 0; i < batch * hidden_size; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f; 
    }
    for (int i = 0; i < hidden_size; i++) {
        h_gamma[i] = 1.0f;  
        h_beta[i] = 0.0f;   
    }

    // allocate GPU memory
    float *d_input, *d_output, *d_gamma, *d_beta;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_gamma, hidden_size * sizeof(float));
    cudaMalloc((void**)&d_beta, hidden_size * sizeof(float));

    // copy data towards GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, hidden_size * sizeof(float), cudaMemcpyHostToDevice);

    //operating Kernel
    launch_layer_norm(d_output, d_input, d_gamma, d_beta, batch, hidden_size);

    // data back towards CPU
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // output
    printf("Input:  [");
    for (int i = 0; i < 10; i++) printf("%.4f ", h_input[i]);
    printf("...]\n");

    printf("Output: [");
    for (int i = 0; i < 10; i++) printf("%.4f ", h_output[i]);
    printf("...]\n");
    printf("Checking LayerNorm Output...\n");
    for (int b = 0; b < batch; b++) {
        printf("Batch %d: ", b);
        compute_mean_std(h_output + b * hidden_size, hidden_size);
    }
    
    // free memory
    free(h_input); free(h_output); free(h_gamma); free(h_beta);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_gamma); cudaFree(d_beta);
}

int main() {
    test_layer_norm();
    return 0;
}
