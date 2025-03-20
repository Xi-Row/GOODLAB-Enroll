__global__ void layer_norm_vec4(float* input, float* output, int seq_len, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = idx; i < seq_len; i += blockDim.x * gridDim.x) {
        float4 vec = *(float4*)&input[i];
        sum.x += vec.x;
        sum.y += vec.y;
        sum.z += vec.z;
        sum.w += vec.w;
    }

    // Reduce within warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum.x += __shfl_down_sync(0xFFFFFFFF, sum.x, offset);
        sum.y += __shfl_down_sync(0xFFFFFFFF, sum.y, offset);
        sum.z += __shfl_down_sync(0xFFFFFFFF, sum.z, offset);
        sum.w += __shfl_down_sync(0xFFFFFFFF, sum.w, offset);
    }

    float mean = (sum.x + sum.y + sum.z + sum.w) / seq_len;

    float4 var_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = idx; i < seq_len; i += blockDim.x * gridDim.x) {
        float4 vec = *(float4*)&input[i];
        float4 diff = make_float4(vec.x - mean, vec.y - mean, vec.z - mean, vec.w - mean);
        var_sum.x += diff.x * diff.x;
        var_sum.y += diff.y * diff.y;
        var_sum.z += diff.z * diff.z;
        var_sum.w += diff.w * diff.w;
    }

    // Reduce within warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        var_sum.x += __shfl_down_sync(0xFFFFFFFF, var_sum.x, offset);
        var_sum.y += __shfl_down_sync(0xFFFFFFFF, var_sum.y, offset);
        var_sum.z += __shfl_down_sync(0xFFFFFFFF, var_sum.z, offset);
        var_sum.w += __shfl_down_sync(0xFFFFFFFF, var_sum.w, offset);
    }

    float variance = (var_sum.x + var_sum.y + var_sum.z + var_sum.w) / seq_len;
    float inv_std = 1.0f / sqrt(variance + epsilon);

    for (int i = idx; i < seq_len; i += blockDim.x * gridDim.x) {
        float4 vec = *(float4*)&input[i];
        float4 diff = make_float4(vec.x - mean, vec.y - mean, vec.z - mean, vec.w - mean);
        *(float4*)&output[i] = make_float4(diff.x * inv_std, diff.y * inv_std, diff.z * inv_std, diff.w * inv_std);
    }
}