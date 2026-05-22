from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .common import N, H_domain, prf, random_bytes, u32
from .lamport_cpu import LamportPublicKey, LamportSecretKey, LamportSignature, LamportOTS


def leaf_hash(leaf_index: int, ots_pk: LamportPublicKey) -> bytes:
    flat = b"".join(a + b for a, b in ots_pk.y)
    return H_domain(b"LMS_LEAF", u32(leaf_index), flat)


def parent_hash(level: int, index: int, left: bytes, right: bytes) -> bytes:
    return H_domain(b"LMS_NODE", u32(level), u32(index), left, right)


@dataclass
class LMSPublicKey:
    root: bytes
    height: int


@dataclass
class LMSSecretKey:
    seed: bytes
    height: int
    next_index: int
    ots_sks: List[LamportSecretKey]
    ots_pks: List[LamportPublicKey]
    tree_levels: List[List[bytes]]


@dataclass
class LMSSignature:
    leaf_index: int
    ots_signature: LamportSignature
    ots_public_key: LamportPublicKey
    auth_path: List[bytes]


class LMS:
    """
    Simplified LMS-style stateful Merkle signature.
      - Uses Lamport OTS leaves instead of full RFC 8554 LM-OTS.
      - Public key is the Merkle root.
      - Signature reveals OTS pk, OTS signature, and auth path.
      - State prevents OTS key reuse.
    """

    scheme_name = "lms"

    @staticmethod
    def _deterministic_lamport_key(seed: bytes, leaf_index: int) -> tuple[LamportPublicKey, LamportSecretKey]:
        x = []
        y = []
        for i in range(256):
            x0 = prf(seed, b"LMS_LAMPORT_X0" + u32(leaf_index), i)
            x1 = prf(seed, b"LMS_LAMPORT_X1" + u32(leaf_index), i)
            y0 = H_domain(b"LAMPORT_PK", i.to_bytes(4, "big"), b"0", x0)
            y1 = H_domain(b"LAMPORT_PK", i.to_bytes(4, "big"), b"1", x1)
            x.append((x0, x1))
            y.append((y0, y1))
        return LamportPublicKey(y=y), LamportSecretKey(x=x)

    @staticmethod
    def _build_tree(leaves: List[bytes]) -> List[List[bytes]]:
        levels = [leaves]
        level = 0
        current = leaves
        while len(current) > 1:
            nxt = []
            for i in range(0, len(current), 2):
                nxt.append(parent_hash(level, i // 2, current[i], current[i + 1]))
            levels.append(nxt)
            current = nxt
            level += 1
        return levels

    @staticmethod
    def keygen(height: int = 4) -> tuple[LMSPublicKey, LMSSecretKey]:
        if height < 1:
            raise ValueError("height must be >= 1")
        if height > 10:
            raise ValueError("height too large for this educational baseline")

        seed = random_bytes(N)
        num_leaves = 1 << height

        ots_pks = []
        ots_sks = []
        leaves = []

        for i in range(num_leaves):
            pk_i, sk_i = LMS._deterministic_lamport_key(seed, i)
            ots_pks.append(pk_i)
            ots_sks.append(sk_i)
            leaves.append(leaf_hash(i, pk_i))

        tree_levels = LMS._build_tree(leaves)
        root = tree_levels[-1][0]

        pk = LMSPublicKey(root=root, height=height)
        sk = LMSSecretKey(seed=seed, height=height, next_index=0, ots_sks=ots_sks, ots_pks=ots_pks, tree_levels=tree_levels)
        return pk, sk

    @staticmethod
    def _auth_path(tree_levels: List[List[bytes]], leaf_index: int) -> List[bytes]:
        auth = []
        idx = leaf_index
        for level_nodes in tree_levels[:-1]:
            auth.append(level_nodes[idx ^ 1])
            idx //= 2
        return auth

    @staticmethod
    def sign(sk: LMSSecretKey, message: bytes) -> LMSSignature:
        num_leaves = 1 << sk.height
        if sk.next_index >= num_leaves:
            raise RuntimeError("LMS private key exhausted: no unused OTS leaves remain")

        idx = sk.next_index
        sk.next_index += 1

        ots_sig = LamportOTS.sign(sk.ots_sks[idx], message)
        auth = LMS._auth_path(sk.tree_levels, idx)

        return LMSSignature(leaf_index=idx, ots_signature=ots_sig, ots_public_key=sk.ots_pks[idx], auth_path=auth)

    @staticmethod
    def verify(pk: LMSPublicKey, message: bytes, sig: LMSSignature) -> bool:
        if sig.leaf_index < 0 or sig.leaf_index >= (1 << pk.height):
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


class LMSCPUBackend:
    def __init__(self, height: int = 4):
        self.height = height
        self.scheme_name = f"lms-h{height}"

    def name(self) -> str:
        return "cpu-python"

    def keygen(self):
        return LMS.keygen(height=self.height)

    def sign(self, sk, message: bytes):
        return LMS.sign(sk, message)

    def verify(self, pk, message: bytes, signature):
        return LMS.verify(pk, message, signature)
