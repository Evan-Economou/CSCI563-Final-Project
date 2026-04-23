# Online Softmax — CUDA Implementation
### Parallel Computing Project | Development Plan
*Spring 2026 | Paper #12: Milakov & Gimelshein, arXiv 2018*

---

## Quick Reference

| | Serial Baseline | Parallel Target | Hardware |
|---|---|---|---|
| **Algorithm** | Safe Softmax (Alg. 2) | Online Softmax (Alg. 3) | Remote cluster GPU |
| **Passes** | 3-pass | 2-pass | CUDA C + Nsight |
| **Mem accesses/elem** | 4 | 3 | |

---

## 1. Project Scope

The serial baseline is **Safe Softmax (Algorithm 2)** — the standard numerically stable 3-pass implementation used by all major DL frameworks. The parallel target is **Online Softmax (Algorithm 3)**, which merges the max and normalization passes into one via a self-correcting running normalizer, reducing memory accesses from 4 to 3 per element.

The key insight enabling parallelism is that the normalizer update operator ⊕ is **associative and commutative**, making it directly expressible as a parallel prefix reduction.

The performance story is straightforward: softmax is memory-bandwidth bound, so fewer passes translate directly into speedup. The paper reports up to **1.3x improvement on a V100**, which serves as the reference number for the final report comparison.

> **Out of scope:** TopK fusion (Algorithm 4) — explicitly excluded to protect time.

---

## 2. Current Status & Risk Assessment

> ⚠️ **STATUS:** Entering Week 3 with no implementation yet. Serial and parallel work must both be completed this week.

Priority order:
1. Serial baseline, fully validated
2. Naive CUDA parallel kernel
3. Incremental optimizations for ablation study

> ⚠️ **RISK:** The project requires 3 performance optimizations with incremental ablation. Plan for 4 optimization variants so one can be dropped if needed.

---

## 3. Weekly Development Plan

### Week 3 — NOW: Serial Implementation + Naive CUDA Kernel

| Task | Owner | Notes / LLM Policy |
|---|---|---|
| Write `safe_softmax_serial.cpp` | Both | **LLM allowed.** Implement Algorithm 2 exactly. Validate on small vectors first. |
| Write correctness tests | Both | **LLM allowed.** Compare output against `scipy.special.softmax` on random vectors. |
| Set up cluster environment | Both | **LLM allowed.** Confirm CUDA toolkit version, module loads, sbatch/srun workflow. |
| Write naive CUDA kernel (v1) | Both | **No LLM.** One threadblock per vector, global memory only. |
| Validate CUDA output vs serial | Both | **No LLM.** Max absolute error must be < 1e-5 before any optimization work. |

### Week 4: Optimizations + Performance Data

| Task | Owner | Notes / LLM Policy |
|---|---|---|
| Optimization 2: CUB reduction | Both | **No LLM.** Replace naive reduction with CUB DeviceReduce over (m,d) pairs. Re-validate. |
| Optimization 3: Shared memory tiling | Both | **No LLM.** Load input tiles into shared memory. Profile with Nsight before and after. |
| Optimization 4: Warp shuffle | Both | **No LLM.** Use `__shfl_down_sync` to eliminate shared mem for intra-warp reduction. |
| Collect speedup data | Both | **No LLM.** Sweep V = 100 to 100,000. Record wall time for each version. |
| Strong scaling analysis | Both | **No LLM.** Fix V, vary thread count. Compute parallel efficiency via Nsight Systems. |
| In-class presentation prep | Both | **LLM allowed for understanding.** 5-min deck: paper, approach, preliminary numbers. |

### Week 5: Analysis, Report, and Presentation

| Task | Owner | Notes / LLM Policy |
|---|---|---|
| Bottleneck analysis with Nsight | Both | **No LLM.** Profile bandwidth utilization, occupancy, SM efficiency per version. |
| Ablation study plot | Both | **No LLM.** Bar chart: Safe serial → v1 → v2 → v3 → v4. Show incremental gains. |
| Compare to paper results | Both | **No LLM.** Report your speedup vs. paper's 1.15x–1.3x. Explain differences. |
| Write final report (3–6 pages) | Both | **No LLM.** Follow Overleaf template. Must be own work. |
| Submit repo + README | Both | Clean code, Makefile, README with build and run instructions. |

---

## 4. Implementation Architecture

### 4.1 File Structure

```
online_softmax/
├── Makefile
├── README.md
├── src/
│   ├── safe_softmax_serial.cpp   # Serial baseline
│   ├── online_softmax_cuda.cu    # All parallel CUDA versions
│   └── utils.h                   # Shared helpers (timer, error check macros)
├── tests/
│   └── correctness_test.cpp      # Validates CUDA output vs serial
└── benchmarks/
    └── benchmark.cu              # Sweeps vector sizes, outputs CSV
```

### 4.2 Serial Baseline — Safe Softmax

Implement Algorithm 2 from the paper exactly:

```cpp
void safe_softmax(const float* x, float* y, int V);
```

Three explicit loops:
1. Find max `m`
2. Accumulate normalizer `d` with `x[j] - m` in the exponent
3. Write `y[i] = exp(x[i] - m) / d`

This is your ground truth for correctness validation — get it right before touching CUDA.

### 4.3 Parallel CUDA — Online Softmax

The parallel version exploits the associativity of the ⊕ operator defined in Section 3.1 of the paper. Each pair `(m_i, d_i)` can be reduced independently and merged:

```
(m_i, d_i) ⊕ (m_j, d_j) = (max(m_i, m_j),  d_i·exp(m_i - max) + d_j·exp(m_j - max))
```

This maps directly onto a standard parallel reduction pattern. Each thread processes a chunk of the input, maintains its own local `(m, d)`, and threads reduce pairwise up a reduction tree. Pass 2 (computing `y_i`) is embarrassingly parallel — one thread per element.

---

## 5. Optimization Roadmap (Ablation Study)

The project requires at least 3 optimizations. Four are planned below, providing one buffer. Each version must be re-validated for correctness before benchmarking.

| # | Optimization | Description | Expected Speedup |
|---|---|---|---|
| 1 | Baseline CUDA kernel | Naive parallel reduction, one thread per element, global memory only | ~1.0x |
| 2 | Parallel reduction with CUB | Use CUB `DeviceReduce` for the (m,d) associative reduction across threads | ~1.1–1.2x |
| 3 | Shared memory tiling | Load tiles into shared memory to reduce global memory latency | ~1.2–1.3x |
| 4 | Warp-level primitives | Replace shared mem reduction with `__shfl_down_sync` for intra-warp passes | ~1.3x |

> **Note:** Each version must maintain max absolute error < 1e-5 vs. serial baseline before it counts as a valid ablation data point.

---

## 6. Performance Analysis Plan

### 6.1 Correctness Test
- Generate random input vectors of size V = {128, 1024, 10000} with values in [-10, 10]
- Compare each CUDA output element against serial: assert `max|y_cuda - y_serial| < 1e-5`
- Run before every benchmark — never benchmark without passing correctness first

### 6.2 Speedup Benchmarks
- Sweep vector size V from 100 to 100,000 (log scale)
- Batch sizes: **10** (latency-limited) and **4,000** (bandwidth-limited), mirroring the paper
- Measure wall time with CUDA events (`cudaEventRecord`), not CPU timers
- 5 warmup iterations, then average over 20 timed iterations
- Plot: elements/second and speedup ratio (Online CUDA / Safe serial) vs. V

### 6.3 Strong Scaling
- Fix V = 10,000, vary thread block size: 32, 64, 128, 256, 512
- Report parallel efficiency = speedup / (threads used)
- Identify where returns diminish — this is a key bottleneck discussion point

### 6.4 Nsight Profiling
- Run Nsight Systems (`nsys profile`) on each optimization version
- Key metrics: DRAM bandwidth utilization, L2 cache hit rate, SM occupancy, warp efficiency
- Compare achieved bandwidth to theoretical peak — quantifies how close to bandwidth-bound the kernel is
- Use Nsight Compute (`ncu`) for kernel-level roofline analysis

---

## 7. LLM Usage Policy Reference

| Activity | Policy |
|---|---|
| Serial implementation (C++) | ✅ Allowed |
| Paper understanding | ✅ Allowed |
| Compile error help (no source) | ✅ Allowed |
| Test generation | ✅ Allowed |
| Parallel CUDA implementation | ❌ Not allowed |
| Performance analysis / debugging | ❌ Not allowed |
| Report writing | ❌ Not allowed |

> **Reminder:** All CUDA kernel code, performance analysis, and the final report must be written without LLM assistance.

---

## 8. Final Report Outline

Target: 3–6 pages using the provided Overleaf template.

1. **Abstract** — what you implemented, key speedup result vs. paper baseline
2. **Paper Summary**
   - Safe Softmax: 3-pass, numerical stability, 4 mem accesses/elem
   - Online Softmax: merged pass, associative ⊕ operator, 3 mem accesses/elem
   - Why fewer memory passes matter on bandwidth-bound hardware
3. **Implementation**
   - Serial: Safe Softmax in C++, validation methodology
   - Parallel: CUDA reduction structure, mapping of ⊕ to threadblock reduction
   - Optimization progression (4 versions)
4. **Performance Results**
   - Ablation study bar chart (v1 → v4)
   - Speedup vs. vector size plot (compare to paper Figures 1 & 2)
   - Strong scaling results
   - Nsight bandwidth utilization numbers
5. **Bottleneck Discussion**
   - At what vector size does DRAM bandwidth become the bottleneck?
   - How does achieved bandwidth compare to peak?
   - Where do you diverge from paper results, and why?
6. **Lessons Learned**

---

## 9. Key Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Serial baseline has subtle bugs | Medium | Validate against scipy on 10+ random vectors before any CUDA work |
| CUDA reduction has race conditions | High for first attempt | Use atomics or CUB first, then optimize. Never optimize before correctness. |
| Cluster job queue delays | Medium | Submit benchmark jobs early; run correctness tests locally if possible |
| Fewer than 3 optimizations show measurable gain | Low if bandwidth-bound | Profile early — if kernel is compute-bound, reconsider tiling strategy |
| Time runs out before report | High given Week 3 start | Collect all performance data in Week 4; reserve Week 5 entirely for writing |
