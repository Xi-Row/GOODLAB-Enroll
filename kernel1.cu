__global__ void LayerNormOne(float* input,float* output,int seq_num,float epsilon) {
    int idx = blockIdx.x + threadIdx.x * blockDim.x;
    if(idx > seq_num)
    return;
    float sum = 0.0f;
    for(int i =0;i < seq_num;i++) {
        sum += input[i];
    }
    float mean = sum / seq_num;
    float var_sum = 0.0f;
    for(int i =0;i<seq_num;i++) {
        float diff = input[i] -mean;
        var_sum += diff * diff;
        }
    float variance = var_sum / seq_num;
    float inv_std = 1.0f / sqrt(variance + epsilon);
    for(int i=0;i<seq_num;i++) {
        output[i] = (input[i] - mean) * inv_std;
    }
    
}