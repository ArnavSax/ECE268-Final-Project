# Hash-Based Digital Signatures: GPU-Accelerated SPHINCS+ and LMS

This repository contains CPU (Python) and GPU (PyCUDA) implementations of several hash-based digital signature schemes:
- **Lamport One-Time Signature (OTS)**
- **Leighton-Micali Signatures (LMS)**: Stateful Merkle tree of Lamport OTS.
- **Simplified SPHINCS+**: Stateless Merkle signature scheme using Lamport OTS.

All GPU operations (leaf generation, public key derivation, Merkle tree construction, and signature verification) are executed on the GPU using custom parallelized CUDA C kernels.

---

## Environment Setup & Prerequisites

To run the GPU-accelerated versions, you need:
1. An NVIDIA GPU with matching NVIDIA CUDA Toolkit installed.
2. Python 3.8+ with PyCUDA installed.

To install PyCUDA in your Python environment:
```bash
pip install pycuda numpy
```

*(Note: The codebase automatically falls back to CPU-only mode if PyCUDA is not installed or if no compatible GPU is detected).*

---

## Run Instructions

### 1. Correctness Verification
To verify the implementation correctness of both CPU and GPU backends, run the unit test suite:
```bash
python -m tests.test_correctness
```
This tests signature generation, valid verification, and tamper/forgery detection for all three schemes.

### 2. Performance Benchmarks
To benchmark the schemes and compare CPU vs GPU performance:
```bash
python -m benchmarks.run_benchmarks --schemes lamport lms sphincs --batches 1 10 --repeats 3 --height 4
```

#### Command-Line Arguments:
- `--schemes`: Schemes to run (choices: `lamport`, `lms`, `sphincs`).
- `--operations`: Operations to benchmark (choices: `keygen`, `sign`, `verify`).
- `--batches`: Batch sizes to benchmark (e.g. `1 10 100`).
- `--repeats`: Number of repeat measurements to run for averaging.
- `--height`: Merkle tree height for LMS and SPHINCS-simple. Note that for LMS, the maximum batch size must be $\le 2^{\text{height}}$ since each stateful signature consumes one one-time signature leaf.
- `--output`: Path to write raw CSV results (defaults to `results/benchmark_results.csv`).

---

## GPU Backend Architecture
The GPU backends are structured under `hashsig/gpu_backend.py`. They interface with the benchmark harness using a unified `SignatureBackend` protocol:
- **`keygen()`**: Builds the Merkle tree entirely on the GPU. The leaves are computed in parallel, and tree levels are hashed level-by-level using a bottom-up grid launch.
- **`sign(sk, message)`**:
  - LMS: Stateful sign. Secret keys are generated on host, and sibling paths are extracted from the host-cached tree.
  - SPHINCS: Stateless sign. Rebuilds the Merkle tree on the GPU to dynamically extract the authentication path.
- **`verify(pk, message, signature)`**: Verifies the Lamport OTS signature component on the GPU using 256 threads (one per bit), then hashes up the path to the root on the GPU to match the public key.

---

## Performance Results

Benchmark measurements on the local environment (mean throughput in ops/sec with tree height $h=4$):

| Scheme | Operation | CPU Backend (ops/s) | GPU Backend (ops/s) | GPU Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Lamport** | keygen | 738.83 | **5066.21** | **6.8x** |
| | sign | 37473.68 | 6297.33 | CPU-bound |
| | verify | 1888.83 | **6793.94** | **3.6x** |
| **LMS-h4** | keygen | 32.22 | **1629.37** | **50.5x** |
| | sign | 30136.33 | 2684.04 | CPU-bound |
| | verify | 1983.13 | **2055.61** | **1.0x** |
| **SPHINCS-simple-h4** | keygen | 33.29 | **1819.26** | **54.6x** |
| | sign | 31.12 | **753.63** | **24.2x** |
| | verify | 1961.63 | **2039.00** | **1.0x** |

