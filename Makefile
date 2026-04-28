CXX       = g++
NVCC      = nvcc
CXXFLAGS  = -O2 -std=c++17 -Wall -Wextra
NVCCFLAGS = -O2 -std=c++14

BUILD  = build
SERIAL = $(BUILD)/safe_softmax_serial
CUDA   = $(BUILD)/online_softmax_parallel

.PHONY: all serial cuda clean test test_cuda

all: serial cuda

serial: $(SERIAL)

cuda: $(CUDA)

$(BUILD):
	mkdir -p $(BUILD)

$(SERIAL): src/safe_softmax_serial.cpp src/util.h | $(BUILD)
	$(CXX) $(CXXFLAGS) src/safe_softmax_serial.cpp -o $@

$(CUDA): src/online_softmax_parallel.cu src/util.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) src/online_softmax_parallel.cu -o $@

test: $(SERIAL)
	python tests/correctness_test.py

test_cuda: $(CUDA)
	python tests/correctness_test.py --cuda

clean:
	rm -rf $(BUILD)
