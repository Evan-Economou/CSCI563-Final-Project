#include <chrono>
#include <limits>
#include <string>
#include "util.h"

// Algorithm 2: Safe Softmax (3-pass)
// Pass 1: find max; Pass 2: accumulate normalizer; Pass 3: write output
static void safe_softmax(const float* x, float* y, int V) {
    // Pass 1: max
    float m = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < V; ++j)
        if (x[j] > m) m = x[j];

    // Pass 2: sum of shifted exponentials
    float d = 0.0f;
    for (int j = 0; j < V; ++j)
        d += std::exp(x[j] - m);

    // Pass 3: normalize
    for (int i = 0; i < V; ++i)
        y[i] = std::exp(x[i] - m) / d;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    // Read input
    std::vector<float> x;
    if (read_file(&x, argv[1])) return 1;

    // Time the kernel (exclude I/O)
    std::vector<float> y(x.size());

    std::printf("Start\n");
    
    auto t0 = std::chrono::high_resolution_clock::now();
    safe_softmax(x.data(), y.data(), static_cast<int>(x.size()));
    auto t1 = std::chrono::high_resolution_clock::now();

    std::printf("Done\n");

    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::fprintf(stderr, "time: %.3f us  (V=%zu)\n", elapsed_us, x.size());

    // Write output
    if(write_out(&y, (argc >= 3) ? argv[2] : nullptr)) return 1;
    
    return 0;
}
