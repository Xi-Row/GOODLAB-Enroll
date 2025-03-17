#include<bits/stdc++.h>
using namespace std;
__global__ void hello_from_gpu() {
    printf("hello world from the GPU\n");}
int main(void) {
    hello_from_gpu<<<4,4>>>();
    cout << "dfadfa";
    int canAccessPeer;
cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
if (canAccessPeer) {
    cudaDeviceEnablePeerAccess(1, 0);
}
    cudaDeviceSynchronize();
    return 0;
}