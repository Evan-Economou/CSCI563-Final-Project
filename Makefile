CXX       = g++
NVCC      = nvcc
CXXFLAGS  = -O2 -std=c++17 -Wall -Wextra
NVCCFLAGS = -O2 -std=c++17

BUILD   = build
SERIAL  = $(BUILD)/safe_softmax_serial
CUDA    = $(BUILD)/online_softmax_parallel
TESTBIN = $(BUILD)/correctness_test

.PHONY: all serial cuda test test_py clean

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

test: $(TESTBIN) $(SERIAL)
	./$(TESTBIN)

test_py: $(SERIAL)
	python tests/correctness_test.py

clean:
	rm -rf $(BUILD)
