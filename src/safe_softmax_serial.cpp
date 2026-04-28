#include <chrono>
#include <limits>
#include <string>
#include "util.h"

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    // Read input
    std::vector<float> x;
    if (read_file(&x, argv[1])) return 1;

    // Time the kernel (exclude I/O)
    std::vector<float> y(x.size());
    
    auto t0 = std::chrono::high_resolution_clock::now();
    safe_softmax(x.data(), y.data(), static_cast<int>(x.size()));
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::fprintf(stderr, "time: %.3f us  (V=%zu)\n", elapsed_us, x.size());

    // Write output
    if(write_out(&y, (argc >= 3) ? argv[2] : nullptr)) return 1;
    
    return 0;
}
