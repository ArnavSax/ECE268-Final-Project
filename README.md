# Hash-Based Digital Signatures CPU Baseline

Educational CPU baseline for an ECE 268 hash-based digital signatures project.

Implemented schemes:

- Lamport one-time signature
- Simplified LMS-style Merkle signature
- Simplified SPHINCS+-style stateless hash-based signature

This repo is intentionally structured so GPU implementations can be plugged into the same benchmark harness later.

## Important note

These implementations are for course benchmarking and correctness experiments only. They are not production-secure cryptographic implementations.

## Run

```bash
python -m tests.test_correctness
python -m benchmarks.run_benchmarks --schemes lamport lms sphincs --batches 1 10 100 --repeats 3
```

Benchmark results are written to `results/benchmark_results.csv`.

## GPU integration

Create a GPU backend class with the same methods as the CPU wrappers:

```python
keygen()
sign(sk, message)
verify(pk, message, signature)
name()
```

Then add it in `benchmarks/run_benchmarks.py`.

For PyCUDA timing, synchronize before stopping the timer:

```python
cuda.Context.synchronize()
```
