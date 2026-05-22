from __future__ import annotations

import hashlib
import os
import struct
import time
from dataclasses import dataclass
from typing import Callable, List

N = 32  # 256-bit toy security parameter / hash output length


def random_bytes(n: int = N) -> bytes:
    return os.urandom(n)


def u32(x: int) -> bytes:
    return struct.pack(">I", x)


def u64(x: int) -> bytes:
    return struct.pack(">Q", x)


def H(data: bytes, out_len: int = N) -> bytes:
    """SHAKE256 variable-length hash."""
    return hashlib.shake_256(data).digest(out_len)


def H_domain(domain: bytes, *parts: bytes, out_len: int = N) -> bytes:
    """Domain-separated hash."""
    buf = bytearray(domain)
    for p in parts:
        buf.extend(u32(len(p)))
        buf.extend(p)
    return H(bytes(buf), out_len=out_len)


def prf(seed: bytes, label: bytes, index: int, out_len: int = N) -> bytes:
    return H_domain(b"PRF", seed, label, u64(index), out_len=out_len)


def message_digest_bits(message: bytes, bits: int = 256) -> List[int]:
    """Return first `bits` bits of SHA256(message), MSB first."""
    digest = hashlib.sha256(message).digest()
    out = []
    for b in digest:
        for i in range(7, -1, -1):
            out.append((b >> i) & 1)
            if len(out) == bits:
                return out
    return out


@dataclass
class BenchmarkResult:
    scheme: str
    backend: str
    operation: str
    batch_size: int
    repeat: int
    elapsed_s: float
    ops_per_s: float

    def to_csv_row(self):
        return [
            self.scheme,
            self.backend,
            self.operation,
            self.batch_size,
            self.repeat,
            f"{self.elapsed_s:.9f}",
            f"{self.ops_per_s:.6f}",
        ]
