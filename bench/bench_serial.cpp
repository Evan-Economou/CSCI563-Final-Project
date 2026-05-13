#include <cstdio>
#include <vector>
#include <chrono>
#include "util.h"

static const int WARMUP = 5;
static const int REPEAT = 20;
static const int SIZES[]  = {100, 316, 1000, 3162, 10000, 31623, 100000, 1000000};
static const int N_SIZES  = 8;

static std::vector<float> make_input(int V) {
    std::vector<float> x(V);
    unsigned s = 42u;
    for (int i = 0; i < V; ++i) {
        s = s * 1664525u + 1013904223u;
        x[i] = ((float)(s >> 1) / (float)0x7FFFFFFFu) * 20.f - 10.f;
    }
    return x;
}

int main() {
    printf("V,safe_us,online_us\n");

    for (int si = 0; si < N_SIZES; ++si) {
        int V = SIZES[si];
        auto x = make_input(V);
        std::vector<float> y(V);

        // --- safe_softmax ---
        for (int r = 0; r < WARMUP; ++r)
            safe_softmax(x.data(), y.data(), V);

        double safe_total = 0.0;
        for (int r = 0; r < REPEAT; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            safe_softmax(x.data(), y.data(), V);
            auto t1 = std::chrono::high_resolution_clock::now();
            safe_total += std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        double safe_us = safe_total / REPEAT;

        // --- online_softmax ---
        for (int r = 0; r < WARMUP; ++r)
            online_softmax(x.data(), y.data(), V);

        double online_total = 0.0;
        for (int r = 0; r < REPEAT; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            online_softmax(x.data(), y.data(), V);
            auto t1 = std::chrono::high_resolution_clock::now();
            online_total += std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        double online_us = online_total / REPEAT;

        printf("%d,%.3f,%.3f\n", V, safe_us, online_us);
    }
    return 0;
}
