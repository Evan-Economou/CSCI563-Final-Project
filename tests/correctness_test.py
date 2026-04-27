"""
Correctness test: compare safe_softmax_serial output against scipy.special.softmax.
Runs the compiled binary on random vectors and asserts max absolute error < 1e-5.
"""

import subprocess
import tempfile
import os
import sys
import numpy as np
from scipy.special import softmax

BINARY = os.path.join(os.path.dirname(__file__), "..", "build", "safe_softmax_serial")
TOL = 1e-5

TEST_SIZES = [4, 128, 1024, 10000]
SEED = 42


def run_binary(x: np.ndarray) -> np.ndarray:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fin:
        fin.write(" ".join(f"{v:.9g}" for v in x))
        fin_path = fin.name
    fout_path = fin_path + ".out"
    try:
        result = subprocess.run(
            [BINARY, fin_path, fout_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
            raise RuntimeError(f"Binary exited with code {result.returncode}")
        # Print timing from stderr
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
    rng = np.random.default_rng(SEED)
    passed = 0
    failed = 0

    for V in TEST_SIZES:
        # Draw from [-10, 10] as specified in the performance plan
        x = rng.uniform(-10.0, 10.0, size=V).astype(np.float32)
        ref = softmax(x.astype(np.float64))  # double-precision reference

        print(f"V={V:>7}: ", end="", flush=True)
        try:
            got = run_binary(x)
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
