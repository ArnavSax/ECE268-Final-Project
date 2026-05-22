# Progress Report Notes / TODOs

## Project scope

The final project is Hash-Based Digital Signatures.

- Lamport one-time signature as warm-up
- LMS and simplified SPHINCS+
- KeyGen, Sign, Verify
- State management for stateful schemes
- Quantify signature size, key size, and per-signature hash count vs ECDSA
- CPU and GPU implementation repo for final submission

## Current CPU baseline status

Implemented:

- Lamport OTS CPU baseline
- Simplified LMS-style CPU baseline using Lamport OTS leaves
- Simplified SPHINCS+-style stateless CPU baseline using deterministic leaf selection
- Correctness tests
- Benchmark harness
- GPU backend stub/API