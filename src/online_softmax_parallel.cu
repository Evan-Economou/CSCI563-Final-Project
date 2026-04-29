#include <cuda_runtime_api.h>
#include <chrono>
#include <limits>
#include <string>
#include "util.h"

#define BLOCK_SIZE      32
#define ELEMS_PER_BLOCK (2 * BLOCK_SIZE)

// Algorithm 3: Online Softmax (3-pass)
__global__ void online_softmax_pass_1(const float* input, float* output, const int size) {
    // Make shared data
    __shared__ float s[ELEMS_PER_BLOCK];
    float* m_shared = s;
    float* d_shared = s + ELEMS_PER_BLOCK/2;


    // Calculate tid
    int tid  = threadIdx.x;
    int base = blockIdx.x * ELEMS_PER_BLOCK;
    int i0   = base + 2 * tid;
    int i1   = base + 2 * tid + 1;
    
    // Pass 1: max and sum of shifted exponentials
    // Pass 1 variables (pad to power of 2)
    if(i0 >= size) { //out of bounds, need padding
        m_shared[tid] = 0;
        d_shared[tid] = 1;
    } else if (i1 >= size) {
        m_shared[tid] = input[i0];
        d_shared[tid] = 1;
    } else {
        float m = input[i0] > input[i1] ? input[i0] : input[i1];
        m_shared[tid] = m;
        d_shared[tid] = std::exp(input[i0] - m) + std::exp(input[i1] - m);
    }

    __syncthreads();
    
    // Rework to halve the remaining block each loop iteration, do calculation, write to shared memory
    for(int i = ELEMS_PER_BLOCK/4; i > 0; i /= 2){

        __syncthreads();
        if (tid < i) {
            //merge the m and d values of tid and tid + i
            float m = m_shared[tid] > m_shared[tid + i] ? m_shared[tid] : m_shared[tid+i];
            m_shared[tid] = m;
            d_shared[tid] = d_shared[tid] * std::exp(m_shared[tid] - m) + d_shared[tid + i] * std::exp(m_shared[tid + i] - m);
        }

    }

    // Write final d and m to main cuda memory, then syncthreads (maybe before as well)
    if(tid == 0) {
        int num_blocks = (size + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
        output[blockIdx.x] = m_shared[0];
        output[num_blocks + blockIdx.x] = d_shared[0];
    }
}

__global__ void online_softmax_global_m_d(float* block_results, float* d_m, const int num_blocks) {
    extern __shared__ float s[];
    float* m_shared = s;
    float* d_shared = s + num_blocks;

    // Calculate tid and populate shared memory
    int tid  = threadIdx.x;
    m_shared[tid] = block_results[tid];
    d_shared[tid] = block_results[tid + num_blocks];

    __syncthreads();

    for(int i = ELEMS_PER_BLOCK/4; i > 0; i /= 2){
        __syncthreads();
        if (tid < i) {
            if(!(tid + i > num_blocks)) {
                //merge the m and d values of tid and tid + i
                float m = m_shared[tid] > m_shared[tid + i] ? m_shared[tid] : m_shared[tid+i];
                m_shared[tid] = m;
                d_shared[tid] = d_shared[tid] * std::exp(m_shared[tid] - m) + d_shared[tid + i] * std::exp(m_shared[tid + i] - m);
            }
            
        }
    }

    if(tid == 0) {
        d_m[0] = d_shared[0];
        d_m[1] = m_shared[0];
    }
}

__global__ void online_softmax_pass_2(const float* input, float* d_m, float* output, const int size) {
    // Calculate tid
    int tid  = threadIdx.x;

    // Pass 2: normalize
    // Standard parallelization, calculate output for each index
    if(tid < size) {
        output[tid] = std::exp(input[tid] - d_m[1]) / d_m[0];
    }
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

    int num_blocks = (size + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    float* m_d_block_output = nullptr;
    cudaMalloc(&m_d_block_output, sizeof(float) * num_blocks * 2);
    
    float* d_m = nullptr; //index 0 is d, index 1 is m
    cudaMalloc(&d_m, sizeof(float) * 2);

    // TODO: use ELEMENT_PER_BLOCK
    
    // Time the kernel (exclude I/O)
    auto t0 = std::chrono::high_resolution_clock::now();
    // CUDA execution 
    online_softmax_pass_1<<<num_blocks, BLOCK_SIZE>>>(d_input, m_d_block_output, size);
    online_softmax_global_m_d<<<1, num_blocks, sizeof(float) * 2 * num_blocks>>>(m_d_block_output, d_m, num_blocks);
    online_softmax_pass_2<<<num_blocks, 2*BLOCK_SIZE>>>(d_input, d_m, d_output, size);
    auto t1 = std::chrono::high_resolution_clock::now();


    // CUDA memory release
    std::vector<float> output(size);
    cudaMemcpy(output.data(), d_output, sizeof(float) * size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(m_d_block_output);

    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::fprintf(stderr, "time: %.3f us  (V=%zu)\n", elapsed_us, input.size());

    // Write output
    if(write_out(&output, (argc >= 3) ? argv[2] : nullptr)) return 1;

    return 0;
}
