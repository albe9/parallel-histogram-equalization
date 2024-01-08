#define SERVER
#include "utils.h"
#include <cuda_runtime.h>

__global__ void printArrayKernel(uint8_t* array, uint32_t idx) {
        
    printf("thread : %d, array[%d] = %d\n", blockIdx.x * blockDim.x + threadIdx.x, idx, array[idx]);
}


void parallel_clache()
{
    // Size of the array
    size_t arraySize = 1000;

    // Host array
    uint8_t* hostArray = new uint8_t[arraySize];
    hostArray[3] = 20;

    // CUDA device array
    uint8_t* deviceArray;

    // Allocate GPU memory
    cudaMalloc((void**)&deviceArray, arraySize * sizeof(uint8_t));

    // Copy data from host to GPU
    cudaMemcpy(deviceArray, hostArray, arraySize * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch the kernel
    printArrayKernel<<<2, 2>>>(deviceArray, 3);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(deviceArray);
}

int main()
{
    parallel_clache();
}