#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep STRIDE=1..100 once per run, repeating the full sweep 5 times."
    )
    parser.add_argument("--source", default="slowest.c", help="C++ source file to compile")
    parser.add_argument("--output", default="result.txt", help="File to write benchmark results")
    parser.add_argument("--binary", default="./a.out", help="Compiled benchmark binary path")
    parser.add_argument("--cxx", default="g++", help="C++ compiler")
    parser.add_argument("--cpu", default="3", help="CPU core for taskset")
    parser.add_argument("--start", type=int, default=1, help="First stride value")
    parser.add_argument("--end", type=int, default=100, help="Last stride value, inclusive")
    parser.add_argument("--runs", type=int, default=5, help="Full stride sweeps to run")
    return parser.parse_args()


def run_checked(cmd):
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        print(f"command failed: {' '.join(error.cmd)}", file=sys.stderr)
        if error.stdout:
            print(error.stdout, file=sys.stderr, end="")
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        raise


def main():
    args = parse_args()

    if args.start <= 0 or args.end < args.start or args.runs <= 0:
        print("invalid range or run count", file=sys.stderr)
        return 2

    source = Path(args.source)
    if not source.exists():
        print(f"source file not found: {source}", file=sys.stderr)
        return 2

    binary = Path(args.binary).resolve()
    result_path = Path(args.output)

    strides = range(args.start, args.end + 1)

    with result_path.open("w", encoding="utf-8") as result_file:
        for run in range(1, args.runs + 1):
            for stride in strides:
                compile_cmd = [
                    args.cxx,
                    "-O3",
                    "-march=native",
                    f"-DSTRIDE={stride}",
                    str(source),
                    "-o",
                    str(binary),
                ]
                print(f"compiling stride={stride}", file=sys.stderr, flush=True)
                run_checked(compile_cmd)

                print(f"run={run} stride={stride}", file=sys.stderr, flush=True)
                result_file.write(f"run={run} ")
                result_file.flush()

                bench_cmd = ["taskset", "-c", args.cpu, str(binary)]
                bench = run_checked(bench_cmd)
                result_file.write(bench.stdout)
                if bench.stderr:
                    print(bench.stderr, file=sys.stderr, end="")
                result_file.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
