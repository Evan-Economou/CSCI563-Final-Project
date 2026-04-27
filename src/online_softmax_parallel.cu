#include <cuda_runtime_api.h>
#include <chrono>
#include <limits>
#include <string>
#include "util.h"

#define BLOCK_SIZE      512
#define ELEMS_PER_BLOCK (2 * BLOCK_SIZE)

// Algorithm 3: Online Softmax (3-pass)
__global__ void online_softmax(const float* x, float* y, int V) {
    // Pass 1: max and sum of shifted exponentials
    float m = -std::numeric_limits<float>::infinity();
    float d = 0.0f;
    for (int j = 0; j < V; ++j)
        float old_m = m;
        if (x[j] > m) m = x[j];
        d = d * std::expf(old_m - m) + std::expf(x[j] - m);

    // Pass 2: normalize
    for (int i = 0; i < V; ++i)
        y[i] = std::expf(x[i] - m) / d;
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <input_file> [output_file]\n"
        "\n"
        "  input_file   — text file with whitespace-separated floats\n"
        "  output_file  — optional; defaults to stdout\n"
        "\n"
        "Timing is always written to stderr.\n", prog);
}

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    // Read input
    std::vector<float> x;
    if (read_file(&x, argv[1])) return 1;

    // CUDA memory allocation and data passing


    // Time the kernel (exclude I/O)
    auto t0 = std::chrono::high_resolution_clock::now();
    // CUDA execution 
    // safe_softmax(x.data(), y.data(), static_cast<int>(x.size()));
    auto t1 = std::chrono::high_resolution_clock::now();


    // CUDA memory release


    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::fprintf(stderr, "time: %.3f us  (V=%zu)\n", elapsed_us, x.size());

    // Do serial computation and correctness comparison
    std::vector<float> y(x.size());
    
    safe_softmax(x.data(), y.data(), static_cast<int>(x.size()));
    //....

    // Write output
    FILE* fout = stdout;
    bool close_out = false;
    if (argc >= 3) {
        fout = std::fopen(argv[2], "w");
        if (!fout) { std::perror(argv[2]); return 1; }
        close_out = true;
    }

    for (std::size_t i = 0; i < y.size(); ++i)
        std::fprintf(fout, "%.9g\n", y[i]);

    if (close_out) std::fclose(fout);
    return 0;
}
