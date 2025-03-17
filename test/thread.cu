#include<stdio.h>
__global__ void prtbythread() {
    int th = threadIdx.x;
    int bl = blockIdx.x;
    int id = th + bl * blockDim.x;
    printf("Thread %d,Block %d,id %d\n",th,bl,id);
}
int main() {
    prtbythread<<<2,4>>>();
    cudaDeviceSynchronize();
    return 0;
}