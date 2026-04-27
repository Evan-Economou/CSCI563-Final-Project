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
SERIAL_BINARY = os.path.join(ROOT, "build", "safe_softmax_serial{}".format(EXE))
CUDA_BINARY   = os.path.join(ROOT, "build", "online_softmax_parallel{}".format(EXE))

TOL        = 1e-5
TEST_SIZES = [4, 128, 1024, 10_000]
SEED       = 42


def run_binary(binary: str, x: np.ndarray) -> np.ndarray:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fin:
        fin.write(" ".join("{:.9g}".format(v) for v in x))
        fin_path = fin.name
    fout_path = fin_path + ".out"
    try:
        result = subprocess.run(
            [binary, fin_path, fout_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("  stderr: {}".format(result.stderr.strip()), file=sys.stderr)
            raise RuntimeError("Binary exited with code {}".format(result.returncode))
        timing = result.stderr.strip()
        if timing:
            print("  {}".format(timing))
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
        print("error: binary not found: {}".format(binary), file=sys.stderr)
        print("  Run 'make {}' first.".format('cuda' if args.cuda else 'serial'), file=sys.stderr)
        sys.exit(1)

    print("Testing {} binary: {}\n".format(label, binary))

    rng    = np.random.default_rng(SEED)
    passed = 0
    failed = 0

    for V in TEST_SIZES:
        x   = rng.uniform(-10.0, 10.0, size=V).astype(np.float32)
        ref = softmax(x.astype(np.float64))

        print("V={:>7}: ".format(V), end="", flush=True)
        try:
            got = run_binary(binary, x)
            err = np.max(np.abs(got - ref))
            status = "PASS" if err < TOL else "FAIL"
            print("max_abs_err={:.3e}  [{}]".format(err, status))
            if status == "PASS":
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print("ERROR — {}".format(e))
            failed += 1

    print("\n{}/{} tests passed".format(passed, passed+failed))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
