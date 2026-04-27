#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <limits>
#include <string>

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
        d += std::expf(x[j] - m);

    // Pass 3: normalize
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
    FILE* fin = std::fopen(argv[1], "r");
    if (!fin) { std::perror(argv[1]); return 1; }

    std::vector<float> x;
    float val;
    while (std::fscanf(fin, "%f", &val) == 1)
        x.push_back(val);
    std::fclose(fin);

    if (x.empty()) {
        std::fprintf(stderr, "error: no floats found in '%s'\n", argv[1]);
        return 1;
    }

    std::vector<float> y(x.size());

    // Time the kernel (exclude I/O)
    auto t0 = std::chrono::high_resolution_clock::now();
    safe_softmax(x.data(), y.data(), static_cast<int>(x.size()));
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::fprintf(stderr, "time: %.3f us  (V=%zu)\n", elapsed_us, x.size());

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
