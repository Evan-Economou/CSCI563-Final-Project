CXX       = g++
NVCC      = nvcc
CXXFLAGS  = -O2 -std=c++17
NVCCFLAGS = -O2 -std=c++14

BUILD   = build
SERIAL  = $(BUILD)/safe_softmax_serial
CUDA    = $(BUILD)/online_softmax_parallel
TESTBIN  = $(BUILD)/correctness_test
BENCHBIN = $(BUILD)/bench_serial

.PHONY: all serial cuda bench_serial test test_cuda test_py clean

all: serial cuda

serial: $(SERIAL)

cuda: $(CUDA)

$(BUILD):
	mkdir -p $(BUILD)

$(SERIAL): src/safe_softmax_serial.cpp src/util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) src/safe_softmax_serial.cpp -o $@

$(CUDA): src/online_softmax_parallel.cu src/util.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) src/online_softmax_parallel.cu -o $@

$(TESTBIN): tests/correctness_test.cpp src/util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc tests/correctness_test.cpp -o $@

$(BENCHBIN): bench/bench_serial.cpp src/util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc bench/bench_serial.cpp -o $@

bench_serial: $(BENCHBIN)

test: $(TESTBIN) $(SERIAL)
	./$(TESTBIN)

test_cuda: $(TESTBIN) $(SERIAL) $(CUDA)
	./$(TESTBIN)

test_py: $(SERIAL)
	python tests/correctness_test.py

clean:
	rm -rf $(BUILD)
	rm -f bench_results.csv tmp_bench_in.txt tmp_bench_out.txt tmp_bench_stderr.txt
