#include <cuda_runtime_api.h>
#include <chrono>
#include <cfloat>
#include <limits>
#include <string>
#include "util.h"
#include "stdio.h"
#include <iostream>

#define BLOCK_SIZE      512
#define ELEMS_PER_BLOCK (2 * BLOCK_SIZE)

__device__  void merge_softmax(float& left_m, float& left_d, float right_m, float right_d) {
    float old_m = left_m;
    float old_d = left_d;
    float m = old_m > right_m ? old_m : right_m;
    left_m = m;
    left_d = old_d * std::exp(old_m - m) + right_d * std::exp(right_m - m);
}

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

    m_shared[tid] = -FLT_MAX;
    d_shared[tid] = 0.0f;
    
    // Pass 1: max and sum of shifted exponentials
    if(i0 < size && i1 < size) {
        float m = input[i0] > input[i1] ? input[i0] : input[i1];
        m_shared[tid] = m;
        d_shared[tid] = std::exp(input[i0] - m) + std::exp(input[i1] - m);
    } else if (i0 < size) {
        m_shared[tid] = input[i0];
        d_shared[tid] = 1;
    }

    __syncthreads();
    
    for(int i = ELEMS_PER_BLOCK/4; i > 0; i /= 2){

        __syncthreads();
        if (tid < i) {
            //merge the m and d values of tid and tid + i
            if (tid + i < ELEMS_PER_BLOCK / 2) {
                merge_softmax(m_shared[tid], d_shared[tid], m_shared[tid + i], d_shared[tid + i]);
            }
        }
    }

    // Write final d and m to main cuda memory, then syncthreads
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
    if (tid < num_blocks) {
        m_shared[tid] = block_results[tid];
        d_shared[tid] = block_results[tid + num_blocks];
    } else {
        m_shared[tid] = -FLT_MAX;
        d_shared[tid] = 0.0f;
    }

    __syncthreads();

    for(int i = ELEMS_PER_BLOCK/4; i > 0; i /= 2){
        __syncthreads();
        if (tid < i) {
            if (tid + i < num_blocks) {
                // merge the m and d values of tid and tid + i
                merge_softmax(m_shared[tid], d_shared[tid], m_shared[tid + i], d_shared[tid + i]);
            }   
        }
    }

    if(tid == 0) {
        d_m[0] = d_shared[0];
        d_m[1] = m_shared[0];
        
    }
}

__global__ void online_softmax_pass_2(const float* input, const float* d_m, float* output, const int size) {
    // Calculate global output index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Pass 2: normalize
    // Standard parallelization, calculate output for each index
    if (tid < size) {
        output[tid] = std::exp(input[tid] - d_m[1]) / d_m[0];
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    // Read input
    std::vector<float> input;
    if (read_file(&input, argv[1])) return 1;
    // for(int i = 0; i< input.size(); i++) {
    //     std::cout << input[i];
    // }
    std::cout << std::endl;

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
    float* blockoutput = new float[num_blocks*2];
    cudaMemcpy(blockoutput, m_d_block_output, sizeof(float) * num_blocks * 2, cudaMemcpyDeviceToHost);
    // for(int i = 0; i < num_blocks; i++) {
    //     std::fprintf(stdout, "Block #%i m: %f \n", i, blockoutput[i]);
    //     std::fprintf(stdout, "Block #%i d: %f \n", i, blockoutput[i + num_blocks]);
    // }
    
    online_softmax_global_m_d<<<1, num_blocks, sizeof(float) * 2 * num_blocks>>>(m_d_block_output, d_m, num_blocks);
    float* local_d_m = new float[2];
    cudaMemcpy(local_d_m, d_m, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    // std::fprintf(stdout, "Final m value: %f \n", local_d_m[1]);
    // std::fprintf(stdout, "Final d value: %f \n", local_d_m[0]);
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
