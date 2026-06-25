#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_PATTERN = "separated_by_stride8_bank_conflicts_and_cacheline"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compile slowest.cc and fetch one perf event for a selected "
            "benchmark access pattern."
        )
    )
    parser.add_argument(
        "event",
        help='perf event to fetch, for example "cpu_core/cache-misses/"',
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"benchmark access pattern to run (default: {DEFAULT_PATTERN})",
    )
    parser.add_argument("--source", default="slowest.cc", help="C++ source file")
    parser.add_argument(
        "--binary",
        default="/tmp/slowest_perf_event",
        help="compiled benchmark binary path",
    )
    parser.add_argument("--cxx", default="g++", help="C++ compiler")
    parser.add_argument("--cpu", default="3", help="CPU core for taskset")
    parser.add_argument(
        "-r", "--repeat", type=int, default=3, help="perf stat repeat count"
    )
    parser.add_argument("--stride", type=int, help="compile with -DSTRIDE=N")
    parser.add_argument(
        "--dram-row-shift", type=int, help="compile with -DDRAM_ROW_SHIFT=N"
    )
    parser.add_argument(
        "--dram-bank-mask",
        help="compile with -DDRAM_BANK_MASK=VALUE, for example 0x3f000",
    )
    parser.add_argument(
        "--dram-bank-xor-shift",
        type=int,
        help="compile with -DDRAM_BANK_XOR_SHIFT=N",
    )
    parser.add_argument(
        "--extra-cxxflag",
        action="append",
        default=[],
        help="extra compiler flag; repeat for multiple flags",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="reuse --binary instead of compiling first",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="print only the fetched counter value",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print compile, perf, and benchmark output to stderr",
    )
    return parser.parse_args()


def run_checked(cmd, verbose=False, stdout=None):
    if verbose:
        print("+ " + " ".join(cmd), file=sys.stderr, flush=True)
    try:
        return subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=stdout if stdout is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        print(f"command failed: {' '.join(error.cmd)}", file=sys.stderr)
        if error.stdout:
            print(error.stdout, file=sys.stderr, end="")
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        raise


def compile_benchmark(args):
    source = Path(args.source)
    if not source.exists():
        print(f"source file not found: {source}", file=sys.stderr)
        return 2

    binary = Path(args.binary)
    binary.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.cxx,
        "-O3",
        "-march=native",
        "-std=c++17",
    ]
    if args.stride is not None:
        cmd.append(f"-DSTRIDE={args.stride}")
    if args.dram_row_shift is not None:
        cmd.append(f"-DDRAM_ROW_SHIFT={args.dram_row_shift}")
    if args.dram_bank_mask is not None:
        cmd.append(f"-DDRAM_BANK_MASK={args.dram_bank_mask}")
    if args.dram_bank_xor_shift is not None:
        cmd.append(f"-DDRAM_BANK_XOR_SHIFT={args.dram_bank_xor_shift}")
    cmd.extend(args.extra_cxxflag)
    cmd.extend([str(source), "-o", str(binary)])

    run_checked(cmd, verbose=args.verbose)
    return 0


def normalize_event_name(event):
    return event.strip().strip("/")


def parse_perf_output(perf_output, requested_event):
    records = []
    requested = normalize_event_name(requested_event)
    for row in csv.reader(perf_output.splitlines()):
        if len(row) < 3:
            continue
        value = row[0].strip()
        unit = row[1].strip()
        event = row[2].strip()
        if not value or not event:
            continue
        if "seconds time elapsed" in event:
            continue
        records.append((value, unit, event, row))

    if not records:
        raise ValueError("perf output did not contain an event counter row")

    matches = [
        record
        for record in records
        if normalize_event_name(record[2]) == requested
        or requested in normalize_event_name(record[2])
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches and len(records) == 1:
        return records[0]
    if matches:
        names = ", ".join(record[2] for record in matches)
        raise ValueError(f"perf output matched multiple event rows: {names}")

    names = ", ".join(record[2] for record in records)
    raise ValueError(f"perf output did not contain {requested_event}; saw: {names}")


def fetch_perf_event(args):
    binary = Path(args.binary)
    if not binary.exists():
        print(f"binary not found: {binary}", file=sys.stderr)
        return 2

    with tempfile.NamedTemporaryFile(prefix="slowest_perf_", delete=False) as file:
        perf_output_path = Path(file.name)

    cmd = [
        "perf",
        "stat",
        "-x",
        ",",
        "-o",
        str(perf_output_path),
        "-r",
        str(args.repeat),
        "-e",
        args.event,
    ]
    if args.cpu:
        cmd.extend(["taskset", "-c", args.cpu])
    cmd.extend([str(binary), args.pattern])

    try:
        result = run_checked(cmd, verbose=args.verbose)
        perf_output = perf_output_path.read_text(encoding="utf-8")
        if args.verbose:
            if result.stdout:
                print(result.stdout, file=sys.stderr, end="")
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="")
            print(perf_output, file=sys.stderr, end="")

        value, unit, event, _row = parse_perf_output(perf_output, args.event)
    finally:
        perf_output_path.unlink(missing_ok=True)

    if args.raw:
        print(value)
    else:
        print(f"pattern={args.pattern}")
        print(f"event={event}")
        if unit:
            print(f"value={value} {unit}")
        else:
            print(f"value={value}")
    return 0


def main():
    args = parse_args()
    if args.repeat <= 0:
        print("--repeat must be positive", file=sys.stderr)
        return 2

    if not args.no_compile:
        compile_status = compile_benchmark(args)
        if compile_status != 0:
            return compile_status

    try:
        return fetch_perf_event(args)
    except (subprocess.CalledProcessError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
