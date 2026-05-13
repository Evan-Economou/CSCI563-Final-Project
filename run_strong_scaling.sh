#!/bin/bash
# Strong scaling study: fix V=1,000,000, vary BLOCK_SIZE = 32, 64, 128, 256, 512.
# Recompiles the CUDA binary for each block size, runs N_RUNS times, averages.
set -e

V=1000000
BLOCK_SIZES=(32 64 128 256 512)
N_RUNS=5
TMP_IN="tmp_ss_in.txt"
TMP_OUT="tmp_ss_out.txt"
TMP_STDERR="tmp_ss_stderr.txt"
CSV="strong_scaling_results.csv"

echo "Generating input ($V elements)..."
python3 -c "
import random
random.seed(42)
with open('$TMP_IN', 'w') as f:
    for _ in range($V):
        f.write(f'{random.uniform(-10.0, 10.0):.6f}\n')
"

echo "V,block_size,threads_per_block,cuda_kernel_us,pass1_us,global_md_us,pass2_us" > "$CSV"

printf "%-12s %14s %14s %12s %12s %13s\n" \
       "block_size" "kernel_total" "pass_1" "global_md" "pass_2" "threads/block"
printf "%-12s %14s %14s %12s %12s %13s\n" \
       "" "(µs avg)" "(µs avg)" "(µs avg)" "(µs avg)" ""
echo "--------------------------------------------------------------------------"

for BS in "${BLOCK_SIZES[@]}"; do
    # Recompile with this block size
    nvcc -O2 -std=c++14 -DBLOCK_SIZE=$BS \
         src/online_softmax_parallel.cu \
         -o build/online_softmax_bs${BS} 2>/dev/null

    total_kernel=0; total_p1=0; total_gmd=0; total_p2=0
    for run in $(seq 1 $N_RUNS); do
        ./build/online_softmax_bs${BS} "$TMP_IN" "$TMP_OUT" 2>"$TMP_STDERR" || true

        t_kernel=$(grep "^time:"      "$TMP_STDERR" | awk '{print $2}')
        t_p1=$(    grep "\[pass_1\]"  "$TMP_STDERR" | awk '{print $2}')
        t_gmd=$(   grep "\[global_md\]" "$TMP_STDERR" | awk '{print $2}')
        t_p2=$(    grep "\[pass_2\]"  "$TMP_STDERR" | awk '{print $2}')

        total_kernel=$(awk -v a="$total_kernel" -v b="$t_kernel" 'BEGIN{printf "%.6f",a+b}')
        total_p1=$(    awk -v a="$total_p1"     -v b="$t_p1"     'BEGIN{printf "%.6f",a+b}')
        total_gmd=$(   awk -v a="$total_gmd"    -v b="$t_gmd"    'BEGIN{printf "%.6f",a+b}')
        total_p2=$(    awk -v a="$total_p2"     -v b="$t_p2"     'BEGIN{printf "%.6f",a+b}')
    done

    avg_kernel=$(awk -v t="$total_kernel" -v n="$N_RUNS" 'BEGIN{printf "%.3f",t/n}')
    avg_p1=$(    awk -v t="$total_p1"     -v n="$N_RUNS" 'BEGIN{printf "%.3f",t/n}')
    avg_gmd=$(   awk -v t="$total_gmd"    -v n="$N_RUNS" 'BEGIN{printf "%.3f",t/n}')
    avg_p2=$(    awk -v t="$total_p2"     -v n="$N_RUNS" 'BEGIN{printf "%.3f",t/n}')

    threads=$((BS * 1))   # threads per block = BLOCK_SIZE (pass_1 uses BLOCK_SIZE threads)

    printf "%-12s %14s %14s %12s %12s %13s\n" \
           "$BS" "$avg_kernel" "$avg_p1" "$avg_gmd" "$avg_p2" "$threads"

    echo "$V,$BS,$threads,$avg_kernel,$avg_p1,$avg_gmd,$avg_p2" >> "$CSV"

    # Remove temp binary
    rm -f build/online_softmax_bs${BS}
done

rm -f "$TMP_IN" "$TMP_OUT" "$TMP_STDERR"
echo ""
echo "Results saved to $CSV"
