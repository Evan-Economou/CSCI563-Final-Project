#!/bin/bash
set -e

INPUT="tmp_1m_input.txt"
OUTPUT="tmp_1m_output.txt"
REPORT="report_1m"

# Generate input file if it doesn't already exist
if [ ! -f "$INPUT" ]; then
    echo "Generating 1,000,000 input elements..."
    python3 -c "
import random
random.seed(42)
with open('$INPUT', 'w') as f:
    for _ in range(1_000_000):
        f.write(f'{random.uniform(-10.0, 10.0):.6f}\n')
"
    echo "Input written to $INPUT"
else
    echo "Using existing $INPUT"
fi

# Profile the binary directly with CUDA + OS runtime tracing
echo "Profiling with nsys..."
nsys profile \
    -t cuda,osrt \
    -o "$REPORT" \
    --force-overwrite true \
    ./build/online_softmax_parallel "$INPUT" "$OUTPUT"

# Generate human-readable stats and save to out.txt
echo "Generating stats..."
nsys stats "${REPORT}.nsys-rep" > out.txt

echo "Done. Stats written to out.txt"
