"""
Correctness test: compare serial/CUDA softmax output against scipy.special.softmax.
Runs the compiled binary on random vectors and asserts max absolute error < 1e-5.

Usage:
  python tests/correctness_test.py           # test serial binary
  python tests/correctness_test.py --cuda    # test CUDA binary
"""

import argparse
import subprocess
import tempfile
import os
import sys
import numpy as np
from scipy.special import softmax

EXE = ".exe" if sys.platform == "win32" else ""

ROOT = os.path.join(os.path.dirname(__file__), "..")
SERIAL_BINARY = os.path.join(ROOT, "build", f"safe_softmax_serial{EXE}")
CUDA_BINARY   = os.path.join(ROOT, "build", f"online_softmax_parallel{EXE}")

TOL        = 1e-5
TEST_SIZES = [4, 128, 1024, 10_000]
SEED       = 42


def run_binary(binary: str, x: np.ndarray) -> np.ndarray:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fin:
        fin.write(" ".join(f"{v:.9g}" for v in x))
        fin_path = fin.name
    fout_path = fin_path + ".out"
    try:
        result = subprocess.run(
            [binary, fin_path, fout_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
            raise RuntimeError(f"Binary exited with code {result.returncode}")
        timing = result.stderr.strip()
        if timing:
            print(f"  {timing}")
        y = np.loadtxt(fout_path, dtype=np.float64)
    finally:
        os.unlink(fin_path)
        if os.path.exists(fout_path):
            os.unlink(fout_path)
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true",
                        help="Test the CUDA binary instead of the serial binary")
    args = parser.parse_args()

    binary = CUDA_BINARY if args.cuda else SERIAL_BINARY
    label  = "CUDA" if args.cuda else "serial"

    if not os.path.isfile(binary):
        print(f"error: binary not found: {binary}", file=sys.stderr)
        print(f"  Run 'make {'cuda' if args.cuda else 'serial'}' first.", file=sys.stderr)
        sys.exit(1)

    print(f"Testing {label} binary: {binary}\n")

    rng    = np.random.default_rng(SEED)
    passed = 0
    failed = 0

    for V in TEST_SIZES:
        x   = rng.uniform(-10.0, 10.0, size=V).astype(np.float32)
        ref = softmax(x.astype(np.float64))

        print(f"V={V:>7}: ", end="", flush=True)
        try:
            got = run_binary(binary, x)
            err = np.max(np.abs(got - ref))
            status = "PASS" if err < TOL else "FAIL"
            print(f"max_abs_err={err:.3e}  [{status}]")
            if status == "PASS":
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR — {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
