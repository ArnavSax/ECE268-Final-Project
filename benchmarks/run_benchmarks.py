from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Dict, List

from hashsig.common import BenchmarkResult
from hashsig.lamport_cpu import LamportCPUBackend
from hashsig.lms_cpu import LMSCPUBackend
from hashsig.sphincs_simple_cpu import SimpleSPHINCSCPUBackend


def make_messages(batch_size: int, message_size: int) -> List[bytes]:
    messages = []
    for i in range(batch_size):
        prefix = f"benchmark message {i} ".encode()
        pad_len = max(0, message_size - len(prefix))
        messages.append(prefix + bytes([i % 256]) * pad_len)
    return messages


def benchmark_operation(backend, operation: str, batch_size: int, message_size: int, repeat: int) -> BenchmarkResult:
    messages = make_messages(batch_size, message_size)

    if operation == "keygen":
        start = time.perf_counter()
        for _ in range(batch_size):
            backend.keygen()
        if "gpu" in backend.name():
            import pycuda.driver as cuda
            cuda.Context.synchronize()
        end = time.perf_counter()

    elif operation == "sign":
        # NOTE: LMS is stateful, so use enough leaves for the requested batch.
        pk, sk = backend.keygen()
        start = time.perf_counter()
        for msg in messages:
            backend.sign(sk, msg)
        if "gpu" in backend.name():
            import pycuda.driver as cuda
            cuda.Context.synchronize()
        end = time.perf_counter()

    elif operation == "verify":
        pk, sk = backend.keygen()
        signatures = [backend.sign(sk, msg) for msg in messages]
        start = time.perf_counter()
        ok = [backend.verify(pk, msg, sig) for msg, sig in zip(messages, signatures)]
        if "gpu" in backend.name():
            import pycuda.driver as cuda
            cuda.Context.synchronize()
        end = time.perf_counter()
        if not all(ok):
            raise RuntimeError(f"verification failed during benchmark for {backend.scheme_name}")

    else:
        raise ValueError(f"unknown operation: {operation}")

    elapsed = end - start
    ops_per_s = batch_size / elapsed if elapsed > 0 else float("inf")

    return BenchmarkResult(
        scheme=backend.scheme_name,
        backend=backend.name(),
        operation=operation,
        batch_size=batch_size,
        repeat=repeat,
        elapsed_s=elapsed,
        ops_per_s=ops_per_s,
    )


def get_backends(selected_schemes: List[str], height: int):
    backends = []

    if "lamport" in selected_schemes:
        backends.append(LamportCPUBackend())
        try:
            from hashsig.gpu_backend import LamportGPUBackend
            backends.append(LamportGPUBackend())
        except ImportError:
            pass

    if "lms" in selected_schemes:
        backends.append(LMSCPUBackend(height=height))
        try:
            from hashsig.gpu_backend import LMSGPUBackend
            backends.append(LMSGPUBackend(height=height))
        except ImportError:
            pass

    if "sphincs" in selected_schemes:
        backends.append(SimpleSPHINCSCPUBackend(height=height))
        try:
            from hashsig.gpu_backend import SimpleSPHINCSGPUBackend
            backends.append(SimpleSPHINCSGPUBackend(height=height))
        except ImportError:
            pass

    return backends


def summarize(results: List[BenchmarkResult]):
    grouped: Dict[tuple, List[BenchmarkResult]] = {}
    for r in results:
        key = (r.scheme, r.backend, r.operation, r.batch_size)
        grouped.setdefault(key, []).append(r)

    print("\nBenchmark summary")
    print("-" * 100)
    print(f"{'scheme':<24} {'backend':<14} {'operation':<10} {'batch':>8} {'mean_s':>12} {'std_s':>12} {'ops/s':>12}")
    print("-" * 100)

    for key, vals in sorted(grouped.items()):
        scheme, backend, op, batch = key
        times = [v.elapsed_s for v in vals]
        throughputs = [v.ops_per_s for v in vals]
        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        mean_ops = statistics.mean(throughputs)
        print(f"{scheme:<24} {backend:<14} {op:<10} {batch:>8} {mean_t:>12.6f} {std_t:>12.6f} {mean_ops:>12.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU/GPU hash signature backends.")
    parser.add_argument("--schemes", nargs="+", default=["lamport", "lms", "sphincs"], choices=["lamport", "lms", "sphincs"])
    parser.add_argument("--operations", nargs="+", default=["keygen", "sign", "verify"], choices=["keygen", "sign", "verify"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--message-size", type=int, default=64)
    parser.add_argument("--height", type=int, default=4, help="Merkle tree height for LMS and simplified SPHINCS. Need 2^height >= max batch for LMS.")
    parser.add_argument("--output", default="results/benchmark_results.csv")
    args = parser.parse_args()

    if max(args.batches) > (1 << args.height) and "lms" in args.schemes:
        raise ValueError("For LMS, max batch size must be <= 2^height because each signature consumes one OTS leaf.")

    backends = get_backends(args.schemes, args.height)
    all_results: List[BenchmarkResult] = []

    for backend in backends:
        for operation in args.operations:
            for batch in args.batches:
                for repeat in range(args.repeats):
                    print(f"Running {backend.scheme_name} {backend.name()} {operation} batch={batch} repeat={repeat}")
                    result = benchmark_operation(
                        backend=backend,
                        operation=operation,
                        batch_size=batch,
                        message_size=args.message_size,
                        repeat=repeat,
                    )
                    all_results.append(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "backend", "operation", "batch_size", "repeat", "elapsed_s", "ops_per_s"])
        for r in all_results:
            writer.writerow(r.to_csv_row())

    summarize(all_results)
    print(f"\nWrote results to {output}")


if __name__ == "__main__":
    main()
