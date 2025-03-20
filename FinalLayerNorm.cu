//A kernel that achieves:
//
//
//
//
#include <cuda_runtime.h>  //including cuda runtime API
#include <stdio.h>

#define WARP_SIZE 32  //A warp containing 32 threads
#define EPSILON 1e-6f //a small number to avoid zero in denominator

// Warp Reduce
__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset); //warp reducing core
    }
    return val;
}

// LayerNorm CUDA Kernel
__global__ void layer_norm_kernel(float* __restrict__ output,      //__restrict__ make the ptr safe to
                                  const float* __restrict__ input, //improve the accessment efficiency
                                  const float* __restrict__ gamma, 
                                  const float* __restrict__ beta, 
                                  int hidden_size) {
    // get message of thread
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane = tid % WARP_SIZE;  // index of thread in warp
    int warp_id = tid / WARP_SIZE;  // thread belongs to which warp 

    //Calculating the number of warps
    int num_warps = blockDim.x / WARP_SIZE;
    int num_vec4 = hidden_size / 4;  // float4 vectorization

    //shared memory
    __shared__ float buffer[2][WARP_SIZE];

    //float4 vectorized access //Disposing 4 float one times
    float4* input4  = (float4*) (input + bid * hidden_size);
    float4* output4 = (float4*) (output + bid * hidden_size);
    float4* gamma4  = (float4*) gamma;
    float4* beta4   = (float4*) beta;

    float sum = 0.0f, sum_sq = 0.0f;
    float4 local_data;

    //Calculate the mean and variance
    for (int i = tid; i < num_vec4; i += blockDim.x) {
        local_data = input4[i];
        float v1 = local_data.x, v2 = local_data.y;
        float v3 = local_data.z, v4 = local_data.w;

        sum += (v1 + v2 + v3 + v4);
        sum_sq += (v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4);
    }

    // Warp Reduce sum
    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    //Shared memory // double-buffered
    if (lane == 0) {
        buffer[0][warp_id] = sum;
        buffer[1][warp_id] = sum_sq;
    }
    __syncthreads();

    //only one warp dealing with the final reduction
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

    //Normalized calculations
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

//operating Kernel
void launch_layer_norm(float* output, const float* input, 
                       const float* gamma, 
                       const float* beta, int batch, 
                       int hidden_size) {
     int threads = 256;  
    int blocks = batch;
    layer_norm_kernel<<<blocks, threads>>>(output, 
                                           input, 
                                           gamma, beta, 
                                           hidden_size);
}
