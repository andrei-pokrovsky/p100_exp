// Microbenchmark to determine how many ports and/or banks in the register file

#include <iostream>

#define N (32 * 1024)
#define NUM_THDS (32 * 8)
#define NUM_REGS (N / NUM_THDS)

// Kernel function to add the elements of three arrays into fourth array
__global__
void add(float *x, float *y, float *z, float *a)
{
    float sums[NUM_REGS];

    for (int i = 0; i < NUM_REGS; ++i) {
        ++a[threadIdx.x + i * blockDim.x];
        int j = index + i * stride;
        sums[i] = a[j];
        sums[i] += x[j];
        sums[i] += y[j];
        sums[i] += z[j];
        a[j] = sums[i];
    }
}

int main(void)
{
    float *x, *y, *z, *a;

    // Allocate Unified Memory – accessible from CPU or GPU
    // FIXME: manually copy memory to and from device
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&z, N*sizeof(float));
    cudaMallocManaged(&a, N*sizeof(float));

    // initialize arrays on the host
    for (int i = 0; i < N; i++) {
        a[i] = 100000.0f;
        x[i] = i * 1.0f;
        y[i] = i * 2.0f;
        z[i] = i * 3.0f;
    }

    // Run kernel on the GPU
    // P100 has 32 * 1024 32-bit registers in an SM
    // 128 registers per thread -> (32 * 1024) / 128 = 256 threads
    add<<<1, NUM_THDS>>>(x, y, z, a);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // print output
    //for (int i = 0; i < N; ++i)
    //    std::cout << a[i] << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(a);

    return 0;
}
