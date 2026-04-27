CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

BUILD    = build
SERIAL   = $(BUILD)/safe_softmax_serial

.PHONY: all clean test

all: $(SERIAL)

$(BUILD):
	mkdir -p $(BUILD)

$(SERIAL): src/safe_softmax_serial.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

test: $(SERIAL)
	python tests/correctness_test.py

clean:
	rm -rf $(BUILD)
