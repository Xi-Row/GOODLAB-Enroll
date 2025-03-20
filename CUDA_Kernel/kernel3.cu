template<int BLOCK_SIZE>
__global__ void layer_norm_double_buffer(float* input, float* output, int seq_len, float epsilon) {
    extern __shared__ float sdata[];
    float* sdata_mean = sdata;
    float* sdata_var = &sdata[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = idx; i < seq_len; i += BLOCK_SIZE) {
        float4 vec = *(float4*)&input[i];
        sum.x += vec.x;
        sum.y += vec.y;
        sum.z += vec.z;
        sum.w += vec.w;
    }

    // Warp reduce
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum.x += __shfl_down_sync(0xFFFFFFFF, sum.x, offset);
        sum.y += __shfl_down_sync(0xFFFFFFFF, sum.y, offset);
        sum.z += __shfl_down_sync(0xFFFFFFFF, sum.z, offset);
        sum.w += __shfl_down_sync(0xFFFFFFFF, sum.w, offset);
    }

    sdata_mean[threadIdx.x] = sum.x + sum.y
}