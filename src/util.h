#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

void safe_softmax(const float* x, float* y, int V) {
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

void online_softmax(const float* input, float* output, int size) {
    // Pass 1: max and sum of shifted exponentials
    float m = input[0];
    float d = 0.0f;
    for (int j = 0; j < size; j++) {
        float old_m = m;
        if (input[j] > m) m = input[j];
        d = d * std::exp(old_m - m) + std::exp(input[j] - m);
    }

    // Pass 2: normalize
    for (int i = 0; i < size; ++i)
        output[i] = std::exp(input[i] - m) / d;
}

static int read_file(std::vector<float>* data, char* fName) {
    FILE* fin = std::fopen(fName, "r");
    if (!fin) { std::perror(fName); return 1; }
    
    float val;
    while (std::fscanf(fin, "%f", &val) == 1)
        data->push_back(val);
    std::fclose(fin);

    if (data->empty()) {
        std::fprintf(stderr, "error: no floats found in '%s'\n", fName);
        return 1;
    }

    return 0;
}

static int write_out(std::vector<float>* data, char* fName) {
    FILE* fout = stdout;
    bool close_out = false;
    if (fName != nullptr) {
        fout = std::fopen(fName, "w");
        if (!fout) { std::perror(fName); return 1; }
        close_out = true;
    }

    for (std::size_t i = 0; i < data->size(); ++i)
        std::fprintf(fout, "%.9g\n", static_cast<double>((*data)[i]));

    if (close_out) std::fclose(fout);

    return 0;
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

#endif
