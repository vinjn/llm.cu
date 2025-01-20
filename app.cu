#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    FILE* fp = fopen("../data/gpt-2.safetensors", "r");
    if (fp == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    // Read the first 10 bytes of the file
    char buffer[10];
    size_t bytesRead = fread(buffer, 1, 10, fp);
    if (bytesRead < 10) {
        printf("Error reading file!\n");
        exit(1);
    }


    int N = 1 << 20; // 1M elements

    // Allocate Unified Memory ¨C accessible from CPU or GPU
    float* x, * y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add << <numBlocks, blockSize >> > (N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}