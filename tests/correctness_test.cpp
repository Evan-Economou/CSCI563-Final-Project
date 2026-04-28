#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
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

// Write input to temp file, call binary, read output back
static bool run_binary(const char* bin, const std::vector<float>& x, std::vector<float>& y_out) {
    const char* in_tmp  = "tmp_corr_in.txt";
    const char* out_tmp = "tmp_corr_out.txt";

    FILE* f = fopen(in_tmp, "w");
    if (!f) { perror(in_tmp); return false; }
    for (float v : x) fprintf(f, "%.9g\n", v);
    fclose(f);

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s %s %s", bin, in_tmp, out_tmp);
    int ret = system(cmd);
    remove(in_tmp);

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

    printf("=== Correctness Test ===\n");
    printf("cuda : %s%s\n\n", CUDA, has_cuda ? "" : "  (not found — skipping)");

    int total = 0, passed = 0;

    for (int si = 0; si < N_SIZES; ++si) {
        int V = SIZES[si];
        auto x = make_input(V, 42u + (unsigned)V);

        // Compute reference output by calling safe_softmax directly
        std::vector<float> ref(V);
        safe_softmax(x.data(), ref.data(), V);

        // Verify reference sums to 1 and is non-negative
        float sum = 0.f;
        bool valid = true;
        for (int i = 0; i < V; ++i) {
            if (ref[i] < 0.f) { valid = false; break; }
            sum += ref[i];
        }
        if (fabsf(sum - 1.0f) > TOL) valid = false;

        printf("V=%-7d  serial(direct): ", V);
        if (valid) {
            printf("PASS  (sum=%.6f)\n", sum);
            ++passed;
        } else {
            printf("FAIL  (sum=%.6f)\n", sum);
        }
        ++total;

        // CUDA binary: compare against serial reference
        if (has_cuda) {
            std::vector<float> y;
            bool ok = run_binary(CUDA, x, y);
            float err = ok ? max_abs_err(ref.data(), y.data(), V) : 1.f;
            bool pass = ok && err < TOL;
            printf("           cuda(bin):   %s  max_err=%.2e\n",
                   pass ? "PASS" : "FAIL", err);
            ++total; if (pass) ++passed;
        }
    }

    printf("\n%d/%d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
