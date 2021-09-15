#include<stdio.h>

__global__ void hello_from_gpu(){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("hello world from block %d and thread %d\n", bid, tid);
}

int main() {
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}