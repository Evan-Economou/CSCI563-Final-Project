#!/bin/bash
# V-sweep benchmark: serial (safe + online) and CUDA kernel timing.
# Outputs bench_results.csv and a human-readable table to stdout.
set -e

SERIAL_BENCH="./build/bench_serial"
CUDA_BIN="./build/online_softmax_parallel"
SIZES=(100 316 1000 3162 10000 31623 100000 1000000)
N_CUDA_RUNS=5            # independent binary runs per size; results are averaged
TMP_IN="tmp_bench_in.txt"
TMP_OUT="tmp_bench_out.txt"
TMP_STDERR="tmp_bench_stderr.txt"
CSV="bench_results.csv"

# ── Build ──────────────────────────────────────────────────────────────────────
echo "Building..."
make bench_serial  2>&1 | tail -1
make cuda          2>&1 | tail -1

# ── Serial sweep ───────────────────────────────────────────────────────────────
echo "Running serial benchmark (warmup=$WARMUP_IGNORED, repeat=20 per size)..."
SERIAL_CSV="$("$SERIAL_BENCH")"
# Format: V,safe_us,online_us  (CSV header on first line)

# ── CUDA sweep ─────────────────────────────────────────────────────────────────
echo "Running CUDA benchmark ($N_CUDA_RUNS runs per size)..."

# Header
echo "V,safe_us,online_us,cuda_kernel_us,speedup_vs_online,cuda_Melem_s" > "$CSV"

echo ""
printf "%-10s %12s %13s %14s %10s %14s\n" \
       "V" "safe_serial" "online_serial" "CUDA_kernel" "speedup" "CUDA_Melem/s"
printf "%-10s %12s %13s %14s %10s %14s\n" \
       "" "(µs)" "(µs)" "(µs)" "(online)" ""
echo "------------------------------------------------------------------------------------"

# Parse serial CSV (skip header) into two associative arrays
declare -A SAFE_US ONLINE_US
while IFS=',' read -r v s o; do
    [[ "$v" == "V" ]] && continue
    SAFE_US[$v]="$s"
    ONLINE_US[$v]="$o"
done <<< "$SERIAL_CSV"

for V in "${SIZES[@]}"; do
    # Generate input file with python3 (same generator used in run_1m.sh)
    python3 -c "
import random
random.seed(42)
with open('$TMP_IN', 'w') as f:
    for _ in range($V):
        f.write(f'{random.uniform(-10.0, 10.0):.6f}\n')
"

    # Run CUDA binary N_CUDA_RUNS times and collect "time: X us" values
    cuda_total=0
    for run in $(seq 1 $N_CUDA_RUNS); do
        "$CUDA_BIN" "$TMP_IN" "$TMP_OUT" 2>"$TMP_STDERR" || true
        t=$(grep "^time:" "$TMP_STDERR" | awk '{print $2}')
        cuda_total=$(awk -v a="$cuda_total" -v b="$t" 'BEGIN{printf "%.6f", a+b}')
    done
    cuda_us=$(awk -v t="$cuda_total" -v n="$N_CUDA_RUNS" 'BEGIN{printf "%.3f", t/n}')

    rm -f "$TMP_IN" "$TMP_OUT" "$TMP_STDERR"

    safe_us="${SAFE_US[$V]}"
    online_us="${ONLINE_US[$V]}"

    # speedup = online_serial / cuda_kernel
    speedup=$(awk -v o="$online_us" -v c="$cuda_us" 'BEGIN{if(c>0) printf "%.2f", o/c; else print "N/A"}')
    # CUDA throughput in Melem/s = V / cuda_us  (elem/µs = Melem/s)
    melem=$(awk -v v="$V" -v c="$cuda_us" 'BEGIN{if(c>0) printf "%.2f", v/c; else print "N/A"}')

    printf "%-10s %12s %13s %14s %10s %14s\n" \
           "$V" "$safe_us" "$online_us" "$cuda_us" "${speedup}x" "$melem"

    echo "$V,$safe_us,$online_us,$cuda_us,$speedup,$melem" >> "$CSV"
done

echo ""
echo "Results saved to $CSV"
