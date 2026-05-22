from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .common import N, H_domain, prf, random_bytes, u32
from .lamport_cpu import LamportPublicKey, LamportSignature, LamportOTS
from .lms_cpu import leaf_hash, parent_hash


@dataclass
class SimpleSPHINCSPublicKey:
    root: bytes
    height: int
    num_leaves: int


@dataclass
class SimpleSPHINCSSecretKey:
    seed: bytes
    height: int
    num_leaves: int


@dataclass
class SimpleSPHINCSSignature:
    leaf_index: int
    ots_public_key: LamportPublicKey
    ots_signature: LamportSignature
    auth_path: List[bytes]


class SimpleSPHINCS:
    """
    Simplified SPHINCS+-style stateless hash-based signature.
      - Stateless signer derives a leaf index from the message and secret seed.
      - Each leaf contains a deterministic Lamport OTS public key.
      - A Merkle tree authenticates all possible leaves.
      - Signature includes OTS pk, OTS signature, and auth path.

    Captures the concepts of:
      - stateless signing
      - tree authentication
      - many hash operations
      - GPU-friendly independent leaf/hash work
    """

    scheme_name = "sphincs-simple"

    @staticmethod
    def _derive_leaf_index(seed: bytes, message: bytes, num_leaves: int) -> int:
        digest = H_domain(b"SPHINCS_LEAF_INDEX", seed, message, out_len=8)
        return int.from_bytes(digest, "big") % num_leaves

    @staticmethod
    def _deterministic_lamport_key(seed: bytes, leaf_index: int):
        x = []
        y = []
        for i in range(256):
            x0 = prf(seed, b"SPHINCS_LAMPORT_X0" + u32(leaf_index), i)
            x1 = prf(seed, b"SPHINCS_LAMPORT_X1" + u32(leaf_index), i)
            y0 = H_domain(b"LAMPORT_PK", i.to_bytes(4, "big"), b"0", x0)
            y1 = H_domain(b"LAMPORT_PK", i.to_bytes(4, "big"), b"1", x1)
            x.append((x0, x1))
            y.append((y0, y1))
        from .lamport_cpu import LamportPublicKey, LamportSecretKey
        return LamportPublicKey(y=y), LamportSecretKey(x=x)

    @staticmethod
    def _build_tree_from_seed(seed: bytes, height: int):
        num_leaves = 1 << height
        leaves = []
        pks = []

        for i in range(num_leaves):
            pk_i, _ = SimpleSPHINCS._deterministic_lamport_key(seed, i)
            pks.append(pk_i)
            leaves.append(leaf_hash(i, pk_i))

        levels = [leaves]
        current = leaves
        level = 0
        while len(current) > 1:
            nxt = []
            for i in range(0, len(current), 2):
                nxt.append(parent_hash(level, i // 2, current[i], current[i + 1]))
            levels.append(nxt)
            current = nxt
            level += 1

        return levels, pks

    @staticmethod
    def keygen(height: int = 4) -> tuple[SimpleSPHINCSPublicKey, SimpleSPHINCSSecretKey]:
        if height < 1:
            raise ValueError("height must be >= 1")
        if height > 10:
            raise ValueError("height too large for this educational baseline")

        seed = random_bytes(N)
        levels, _ = SimpleSPHINCS._build_tree_from_seed(seed, height)
        root = levels[-1][0]
        num_leaves = 1 << height

        return (
            SimpleSPHINCSPublicKey(root=root, height=height, num_leaves=num_leaves),
            SimpleSPHINCSSecretKey(seed=seed, height=height, num_leaves=num_leaves),
        )

    @staticmethod
    def _auth_path(levels: List[List[bytes]], leaf_index: int) -> List[bytes]:
        auth = []
        idx = leaf_index
        for level_nodes in levels[:-1]:
            auth.append(level_nodes[idx ^ 1])
            idx //= 2
        return auth

    @staticmethod
    def sign(sk: SimpleSPHINCSSecretKey, message: bytes) -> SimpleSPHINCSSignature:
        leaf_index = SimpleSPHINCS._derive_leaf_index(sk.seed, message, sk.num_leaves)
        levels, _ = SimpleSPHINCS._build_tree_from_seed(sk.seed, sk.height)

        ots_pk, ots_sk = SimpleSPHINCS._deterministic_lamport_key(sk.seed, leaf_index)
        ots_sig = LamportOTS.sign(ots_sk, message)
        auth = SimpleSPHINCS._auth_path(levels, leaf_index)

        return SimpleSPHINCSSignature(leaf_index=leaf_index, ots_public_key=ots_pk, ots_signature=ots_sig, auth_path=auth)

    @staticmethod
    def verify(pk: SimpleSPHINCSPublicKey, message: bytes, sig: SimpleSPHINCSSignature) -> bool:
        if sig.leaf_index < 0 or sig.leaf_index >= pk.num_leaves:
            return False
        if len(sig.auth_path) != pk.height:
            return False
        if not LamportOTS.verify(sig.ots_public_key, message, sig.ots_signature):
            return False

        node = leaf_hash(sig.leaf_index, sig.ots_public_key)
        idx = sig.leaf_index

        for level, sibling in enumerate(sig.auth_path):
            if idx % 2 == 0:
                node = parent_hash(level, idx // 2, node, sibling)
            else:
                node = parent_hash(level, idx // 2, sibling, node)
            idx //= 2

        return node == pk.root


class SimpleSPHINCSCPUBackend:
    def __init__(self, height: int = 4):
        self.height = height
        self.scheme_name = f"sphincs-simple-h{height}"

    def name(self) -> str:
        return "cpu-python"

    def keygen(self):
        return SimpleSPHINCS.keygen(height=self.height)

    def sign(self, sk, message: bytes):
        return SimpleSPHINCS.sign(sk, message)

    def verify(self, pk, message: bytes, signature):
        return SimpleSPHINCS.verify(pk, message, signature)
