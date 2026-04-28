#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include "util.h"

#ifdef _WIN32
  #define EXE_EXT ".exe"
#else
  #define EXE_EXT ""
#endif

static const float TOL     = 1e-5f;
static const char* CUDA    = "build/online_softmax_parallel" EXE_EXT;
static const int   SIZES[] = { 4, 128, 1024, 10000 };
static const int   N_SIZES = 4;

// Deterministic LCG — reproducible inputs in [-10, 10]
static std::vector<float> make_input(int V, unsigned seed) {
    std::vector<float> x(V);
    unsigned s = seed;
    for (int i = 0; i < V; ++i) {
        s = s * 1664525u + 1013904223u;
        x[i] = ((float)(s >> 1) / (float)0x7FFFFFFFu) * 20.f - 10.f;
    }
    return x;
}

static float max_abs_err(const float* a, const float* b, int n) {
    float e = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > e) e = d;
    }
    return e;
}

static bool file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f) { fclose(f); return true; }
    return false;
}

// Run CUDA binary, capture output values and parse kernel time from its stderr.
// time_us is set to -1 if the timing line is not found.
static bool run_binary(const char* bin, const std::vector<float>& x,
                       std::vector<float>& y_out, double& time_us) {
    const char* in_tmp   = "tmp_corr_in.txt";
    const char* out_tmp  = "tmp_corr_out.txt";
    const char* time_tmp = "tmp_corr_time.txt";
    time_us = -1.0;

    FILE* f = fopen(in_tmp, "w");
    if (!f) { perror(in_tmp); return false; }
    for (float v : x) fprintf(f, "%.9g\n", v);
    fclose(f);

    // Redirect stderr to a file so we can parse the kernel timing
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s %s %s 2>%s", bin, in_tmp, out_tmp, time_tmp);
    int ret = system(cmd);
    remove(in_tmp);

    // Parse "time: X.XXX us  (V=Y)" from captured stderr
    FILE* tf = fopen(time_tmp, "r");
    if (tf) {
        char line[128];
        while (fgets(line, sizeof(line), tf)) {
            double t;
            if (sscanf(line, "time: %lf us", &t) == 1) { time_us = t; break; }
        }
        fclose(tf);
        remove(time_tmp);
    }

    if (ret != 0) {
        fprintf(stderr, "    [binary exited with code %d]\n", ret);
        remove(out_tmp);
        return false;
    }

    y_out.clear();
    bool err = read_file(&y_out, (char*)out_tmp);
    remove(out_tmp);
    if (err) return false;

    if ((int)y_out.size() != (int)x.size()) {
        fprintf(stderr, "    [expected %d values, got %zu]\n", (int)x.size(), y_out.size());
        return false;
    }
    return true;
}

int main() {
    bool has_cuda = file_exists(CUDA);

    printf("=== Correctness & Timing Test ===\n");
    printf("cuda : %s%s\n\n", CUDA, has_cuda ? "" : "  (not found — skipping)");

    int total = 0, passed = 0;

    for (int si = 0; si < N_SIZES; ++si) {
        int V = SIZES[si];
        auto x = make_input(V, 42u + (unsigned)V);

        printf("V=%-7d\n", V);

        // --- Serial: call safe_softmax() directly and time just the kernel ---
        std::vector<float> ref(V);
        auto t0 = std::chrono::high_resolution_clock::now();
        safe_softmax(x.data(), ref.data(), V);
        auto t1 = std::chrono::high_resolution_clock::now();
        double serial_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        float sum = 0.f;
        bool serial_ok = true;
        for (int i = 0; i < V; ++i) {
            if (ref[i] < 0.f) { serial_ok = false; break; }
            sum += ref[i];
        }
        if (fabsf(sum - 1.0f) > TOL) serial_ok = false;

        printf("  safe (serial) : %s  time=%.3f us\n", serial_ok ? "PASS" : "FAIL", serial_us);
        ++total; if (serial_ok) ++passed;

        // --- Serial online softmax: check against the safe reference ---
        std::vector<float> online(V);
        auto t2 = std::chrono::high_resolution_clock::now();
        online_softmax(x.data(), online.data(), V);
        auto t3 = std::chrono::high_resolution_clock::now();
        double online_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

        float online_err = max_abs_err(ref.data(), online.data(), V);
        bool online_ok = online_err < TOL;

        printf("  online (serial): %s  time=%.3f us  same=%s (max_err=%.2e)\n",
               online_ok ? "PASS" : "FAIL",
               online_us,
               online_ok ? "YES" : "NO",
               online_err);
        ++total; if (online_ok) ++passed;

        // --- CUDA: run binary, get kernel time from its stderr ---
        if (has_cuda) {
            std::vector<float> y;
            double cuda_us;
            bool ok = run_binary(CUDA, x, y, cuda_us);
            float err = ok ? max_abs_err(ref.data(), y.data(), V) : 1.f;
            bool pass = ok && err < TOL;

            printf("  online (parallel)   : %s  ", pass ? "PASS" : "FAIL");
            if (cuda_us >= 0.0) printf("time=%.3f us  ", cuda_us);
            else                printf("time=N/A       ");
            printf("same=%s (max_err=%.2e)\n", pass ? "YES" : "NO", err);
            ++total; if (pass) ++passed;
        }
    }

    printf("\n%d/%d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
