#include <stdio.h>
#include <stdlib.h>

// 矩阵乘法核函数
__global__ void matrixMul(float *C, float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(void) {
    int N = 1024; // 矩阵大小 N x N
    size_t size = N * N * sizeof(float);
    cudaSetDevice(0); // 设置当前设备为 GPU 0
    // 在主机上分配内存
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // 初始化矩阵 A 和 B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f; // 简单起见，所有元素设为 1
        h_B[i] = 1.0f;
    }

    // 在设备上分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义线程块和网格的大小
    dim3 threadsPerBlock(16, 16); // 每个线程块有 16x16 个线程
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 启动核函数
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, N);

    // 检查核函数是否启动成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 等待核函数执行完成
    cudaDeviceSynchronize();

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果（简单检查）
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (h_C[i] != N) {
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Matrix multiplication result is correct.\n");
    } else {
        printf("Matrix multiplication result is incorrect.\n");
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}