from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .common import N, H_domain, message_digest_bits, random_bytes


@dataclass
class LamportPublicKey:
    y: List[Tuple[bytes, bytes]]  # 256 pairs of public hash values


@dataclass
class LamportSecretKey:
    x: List[Tuple[bytes, bytes]]  # 256 pairs of secret random values


@dataclass
class LamportSignature:
    values: List[bytes]  # one secret value per message digest bit


class LamportOTS:
    """
    Lamport one-time signature.

    For a 256-bit message digest:
      - secret key: 256 pairs of random n-byte values
      - public key: hashes of each secret value
      - signature: one selected secret from each pair
    """

    scheme_name = "lamport"

    @staticmethod
    def keygen() -> tuple[LamportPublicKey, LamportSecretKey]:
        x = []
        y = []
        for i in range(256):
            x0 = random_bytes(N)
            x1 = random_bytes(N)
            y0 = H_domain(b"LAMPORT_PK", i.to_bytes(4, "big"), b"0", x0)
            y1 = H_domain(b"LAMPORT_PK", i.to_bytes(4, "big"), b"1", x1)
            x.append((x0, x1))
            y.append((y0, y1))
        return LamportPublicKey(y=y), LamportSecretKey(x=x)

    @staticmethod
    def sign(sk: LamportSecretKey, message: bytes) -> LamportSignature:
        bits = message_digest_bits(message, bits=256)
        sig = [sk.x[i][bit] for i, bit in enumerate(bits)]
        return LamportSignature(values=sig)

    @staticmethod
    def verify(pk: LamportPublicKey, message: bytes, sig: LamportSignature) -> bool:
        if len(sig.values) != 256:
            return False

        bits = message_digest_bits(message, bits=256)

        for i, bit in enumerate(bits):
            candidate = H_domain(
                b"LAMPORT_PK",
                i.to_bytes(4, "big"),
                b"1" if bit else b"0",
                sig.values[i],
            )
            if candidate != pk.y[i][bit]:
                return False
        return True


class LamportCPUBackend:
    scheme_name = "lamport"

    def name(self) -> str:
        return "cpu-python"

    def keygen(self):
        return LamportOTS.keygen()

    def sign(self, sk, message: bytes):
        return LamportOTS.sign(sk, message)

    def verify(self, pk, message: bytes, signature):
        return LamportOTS.verify(pk, message, signature)
