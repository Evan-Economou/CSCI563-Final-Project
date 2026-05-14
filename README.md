# CSCI563 Final Project — Online Softmax (CUDA)

Compares a numerically stable 3-pass serial softmax baseline against a 2-pass online softmax algorithm (Milakov & Gimelshein, 2018) implemented in CUDA. The online algorithm merges the max and sum passes via an associative running normalizer, reducing memory bandwidth per element.

## Codebase Map

```
src/
  util.h                       - shared utilities: safe_softmax(), online_softmax(), file I/O
  safe_softmax_serial.cpp      - serial 3-pass baseline; reads input file, writes output, reports timing
  online_softmax_parallel.cu   - CUDA 2-pass kernel

bench/
  bench_serial.cpp             - CPU benchmark: times safe_softmax vs online_softmax across 8 sizes (100–1M elements), 5 warmup + 20 timed reps, prints CSV to stdout

tests/
  correctness_test.cpp         - C++ validator: compares serial and CUDA outputs against reference
  correctness_test.py          - Python validator: compares binary outputs against scipy.special.softmax

run_bench.sh                   - benchmark across 8 sizes and outputs bench_results.csv
run_strong_scaling.sh          - recompiles and runs kernel at BLOCK_SIZE={32,64,128,256,512} and outputs strong_scaling_results.csv
run_nsys.sh                    - profiles the CUDA binary with Nsight and outputs report_1m.nsys-rep and out.txt
run_1m.sh                      - quick test on 1,000,000 elements

Makefile                       - build targets: serial, cuda, bench_serial, test, test_cuda, test_py, clean
```

## Build & Run

```bash
make serial       # build safe_softmax_serial.exe
make cuda         # build online_softmax_parallel.exe
make test         # run C++ correctness tests
make test_py      # run Python/scipy correctness tests
make test_cuda    # run tests for the CUDA implementation
make clean        # remove build/
```

Binaries take an input file of whitespace-separated floats and an output file as arguments. Timing (excluding I/O) is written to stderr.
