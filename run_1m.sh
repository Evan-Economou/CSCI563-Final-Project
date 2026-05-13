#!/bin/bash
set -e

INPUT="tmp_1m_input.txt"
OUTPUT="tmp_1m_output.txt"

echo "Generating 1,000,000 input elements..."
python3 -c "
import random
random.seed(42)
with open('$INPUT', 'w') as f:
    for _ in range(1_000_000):
        f.write(f'{random.uniform(-10.0, 10.0):.6f}\n')
"

echo "Running online_softmax_parallel..."
./build/online_softmax_parallel "$INPUT" "$OUTPUT"

echo "Output written to $OUTPUT"
rm -f "$INPUT"
