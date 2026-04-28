#include <cuda_runtime_api.h>
#include <chrono>
#include <limits>
#include <string>
#include "util.h"

#define BLOCK_SIZE      32
#define ELEMS_PER_BLOCK (2 * BLOCK_SIZE)

// Algorithm 3: Online Softmax (3-pass)
__global__ void online_softmax(const float* input, float* output, int size) {
    // Make shared data
    __shared__ int s[ELEMS_PER_BLOCK];

    // Calculate tid
    int tid  = threadIdx.x;
    int base = blockIdx.x * ELEMS_PER_BLOCK;
    int i0   = base + 2 * tid;
    int i1   = base + 2 * tid + 1;
    
    // Pass 1: max and sum of shifted exponentials
    // Pass 1 variables (pad to power of 2)
    float m = input[0];
    float d = 0.0f;

    // Rework to halve the remaining block each loop iteration, do calculation, write to shared memory
    for (int j = 0; j < size; j++) {
        float old_m = m;
        if (input[j] > m) m = input[j];
        d = d * std::exp(old_m - m) + std::exp(input[j] - m);
    }

    // Write final d and m to main cuda memory, then syncthreads (maybe before as well)

    // Pass 2: normalize
    // Standard parallelization, calculate output for each index
    for (int i = 0; i < size; ++i)
        output[i] = std::exp(input[i] - m) / d;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    // Read input
    std::vector<float> input;
    if (read_file(&input, argv[1])) return 1;

    int size = (int)input.size();

    // CUDA memory allocation and data passing
    float* d_input = nullptr;
    cudaMalloc(&d_input, sizeof(float) * size);
    cudaMemcpy(d_input, input.data(), sizeof(float) * size, cudaMemcpyHostToDevice);

    float* d_output = nullptr;
    cudaMalloc(&d_output, sizeof(float) * size);
    
    // int* d_size = nullptr;
    // cudaMalloc(&d_size, sizeof(int));
    // cudaMemSet(d_size, size, sizeof(int));

    // TODO: use ELEMENT_PER_BLOCK
    int num_blocks = (size + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    // Time the kernel (exclude I/O)
    auto t0 = std::chrono::high_resolution_clock::now();
    // CUDA execution 
    online_softmax<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, size);
    auto t1 = std::chrono::high_resolution_clock::now();


    // CUDA memory release
    float* output = new float[size];
    cudaMemcpy(output, d_output, sizeof(float) * size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::fprintf(stderr, "time: %.3f us  (V=%zu)\n", elapsed_us, input.size());

    // Do serial computation and correctness comparison
    
    
    safe_softmax(input.data(), output, static_cast<int>(input.size()));
    //....

    // Write output
    std::vector<float> temp_output(output, output+size);
    if(write_out(&temp_output, (argc >= 3) ? argv[2] : nullptr)) return 1;

    return 0;
}
