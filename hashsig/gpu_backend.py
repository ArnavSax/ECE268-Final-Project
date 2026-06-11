# hashsig/gpu_backend.py
from __future__ import annotations

import os
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import hashlib
from typing import Tuple

# Compiled module cache to avoid recompilation overhead
_mod = None

def _get_function(name: str):
    global _mod
    if _mod is None:
        from .cuda_kernel import cuda_code
        _mod = SourceModule(cuda_code)
    return _mod.get_function(name)


class LamportGPUBackend:
    scheme_name = "lamport"

    def name(self) -> str:
        return "gpu-pycuda"

    def keygen(self) -> Tuple[np.ndarray, np.ndarray]:
        # Generate private key on host
        sk = np.random.bytes(512 * 32)
        sk_numpy = np.frombuffer(sk, dtype=np.uint8).copy()
        
        # Allocate device memory
        sk_device = cuda.mem_alloc(sk_numpy.nbytes)
        pk_device = cuda.mem_alloc(sk_numpy.nbytes)
        
        # Copy to device
        cuda.memcpy_htod(sk_device, sk_numpy)
        
        # Run public key generation kernel
        generate_lamport_pk = _get_function("generate_lamport_pk")
        generate_lamport_pk(sk_device, pk_device, block=(512, 1, 1), grid=(1, 1))
        
        # Copy PK back
        pk_numpy = np.zeros_like(sk_numpy)
        cuda.memcpy_dtoh(pk_numpy, pk_device)
        
        return pk_numpy, sk_numpy

    def sign(self, sk: np.ndarray, message: bytes) -> np.ndarray:
        msg_hash = hashlib.sha256(message).digest()
        msg_bits = np.unpackbits(np.frombuffer(msg_hash, dtype=np.uint8))
        
        signature = np.zeros((256, 32), dtype=np.uint8)
        for i, bit in enumerate(msg_bits):
            sk_idx = (i * 2) + int(bit)
            signature[i] = sk[sk_idx * 32 : (sk_idx + 1) * 32]
        return signature.flatten()

    def verify(self, pk: np.ndarray, message: bytes, signature: np.ndarray) -> bool:
        msg_hash = hashlib.sha256(message).digest()
        
        # Allocate device memory
        sig_device = cuda.mem_alloc(signature.nbytes)
        pk_device = cuda.mem_alloc(pk.nbytes)
        msg_hash_device = cuda.mem_alloc(len(msg_hash))
        is_valid_host = np.array([1], dtype=np.int32)
        is_valid_device = cuda.mem_alloc(is_valid_host.nbytes)
        
        # Copy to device
        cuda.memcpy_htod(sig_device, signature)
        cuda.memcpy_htod(pk_device, pk)
        cuda.memcpy_htod(msg_hash_device, msg_hash)
        cuda.memcpy_htod(is_valid_device, is_valid_host)
        
        # Run verify kernel
        verify_lamport_sig = _get_function("verify_lamport_sig")
        verify_lamport_sig(
            sig_device, pk_device, msg_hash_device, is_valid_device,
            block=(256, 1, 1), grid=(1, 1)
        )
        
        # Copy back validation result
        cuda.memcpy_dtoh(is_valid_host, is_valid_device)
        return is_valid_host[0] == 1


class LMSGPUSecretKey:
    def __init__(self, seed: bytes, height: int, public_keys_host: np.ndarray, tree_host: np.ndarray):
        self.seed = seed
        self.height = height
        self.q = 0
        self.public_keys_host = public_keys_host
        self.tree_host = tree_host


class LMSGPUBackend:
    def __init__(self, height: int = 4):
        self.height = height
        self.scheme_name = f"lms-h{height}"

    def name(self) -> str:
        return "gpu-pycuda"

    def keygen(self) -> Tuple[bytes, LMSGPUSecretKey]:
        num_leaves = 1 << self.height
        seed = os.urandom(32)
        
        # Allocate GPU memory
        seed_device = cuda.mem_alloc(32)
        cuda.memcpy_htod(seed_device, seed)
        
        public_keys_bytes = num_leaves * 16384
        public_keys_device = cuda.mem_alloc(public_keys_bytes)
        
        tree_bytes = (1 << (self.height + 1)) * 32
        lms_tree_device = cuda.mem_alloc(tree_bytes)
        
        # Initialize tree memory to zero
        zero_tree = np.zeros(tree_bytes, dtype=np.uint8)
        cuda.memcpy_htod(lms_tree_device, zero_tree)
        
        # Generate leaf OTS public keys
        generate_lms_leaf_pks = _get_function("generate_lms_leaf_pks")
        generate_lms_leaf_pks(
            seed_device, public_keys_device,
            block=(512, 1, 1), grid=(num_leaves, 1)
        )
        
        # Hash leaves onto Merkle tree
        hash_lms_leaves = _get_function("hash_lms_leaves")
        threads_per_block = 256
        grid_size = int((num_leaves + threads_per_block - 1) / threads_per_block)
        hash_lms_leaves(
            public_keys_device, lms_tree_device, np.int32(num_leaves),
            block=(threads_per_block, 1, 1), grid=(grid_size, 1)
        )
        
        # Construct tree levels
        build_lms_tree_level = _get_function("build_lms_tree_level")
        for level in range(self.height - 1, -1, -1):
            num_nodes = 1 << level
            level_nodes_start = 1 << level
            
            block_size = min(num_nodes, 256)
            grid_size = int((num_nodes + block_size - 1) / block_size)
            
            build_lms_tree_level(
                lms_tree_device, np.int32(level_nodes_start), np.int32(num_nodes),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            
        # Copy tree and public keys to host to store in SecretKey
        tree_host = np.zeros(tree_bytes, dtype=np.uint8)
        cuda.memcpy_dtoh(tree_host, lms_tree_device)
        
        public_keys_host = np.zeros(public_keys_bytes, dtype=np.uint8)
        cuda.memcpy_dtoh(public_keys_host, public_keys_device)
        
        root = tree_host[32:64].tobytes()
        pk = self.height.to_bytes(4, 'big') + root
        sk = LMSGPUSecretKey(seed, self.height, public_keys_host, tree_host)
        
        return pk, sk

    def sign(self, sk: LMSGPUSecretKey, message: bytes) -> bytes:
        num_leaves = 1 << sk.height
        if sk.q >= num_leaves:
            raise ValueError("All OTS keys in this LMS tree have been used!")
            
        leaf_q = sk.q
        sk.q += 1
        
        # Generate private key for leaf_q on host
        sk_leaves = []
        for i in range(512):
            inp = sk.seed + leaf_q.to_bytes(4, 'big') + i.to_bytes(2, 'big')
            sk_leaves.append(hashlib.sha256(inp).digest())
            
        # Hash message and build signature
        msg_hash = hashlib.sha256(message).digest()
        msg_bits = np.unpackbits(np.frombuffer(msg_hash, dtype=np.uint8))
        
        sig_components = []
        for i, bit in enumerate(msg_bits):
            sk_idx = i * 2 + int(bit)
            sig_components.append(sk_leaves[sk_idx])
        signature_bytes = b"".join(sig_components)
        
        # Retrieve Lamport Public Key for leaf_q
        offset = leaf_q * 16384
        public_key_q_bytes = sk.public_keys_host[offset : offset + 16384].tobytes()
        
        # Construct authentication path
        path = []
        node_num = (1 << sk.height) + leaf_q
        while node_num > 1:
            sibling = (node_num - 1) if (node_num % 2 == 1) else (node_num + 1)
            sibling_hash = sk.tree_host[sibling * 32 : (sibling + 1) * 32]
            path.append(sibling_hash)
            node_num //= 2
            
        path_bytes = b"".join(path)
        
        lms_sig = leaf_q.to_bytes(4, 'big') + signature_bytes + public_key_q_bytes + path_bytes
        return lms_sig

    def verify(self, pk: bytes, message: bytes, signature: bytes) -> bool:
        # Parse LMS PubKey
        h = int.from_bytes(pk[:4], 'big')
        expected_root = pk[4:36]
        
        # Parse Signature
        q = int.from_bytes(signature[:4], 'big')
        lamport_sig = signature[4:8196]
        lamport_pk = signature[8196:24580]
        path = signature[24580:]
        
        if len(lamport_sig) != 8192 or len(lamport_pk) != 16384 or len(path) != h * 32:
            return False
            
        msg_hash = hashlib.sha256(message).digest()
        
        # GPU allocations
        lamport_sig_device = cuda.mem_alloc(8192)
        lamport_pk_device = cuda.mem_alloc(16384)
        msg_hash_device = cuda.mem_alloc(32)
        is_valid_host = np.array([1], dtype=np.int32)
        is_valid_device = cuda.mem_alloc(4)
        
        cuda.memcpy_htod(lamport_sig_device, lamport_sig)
        cuda.memcpy_htod(lamport_pk_device, lamport_pk)
        cuda.memcpy_htod(msg_hash_device, msg_hash)
        cuda.memcpy_htod(is_valid_device, is_valid_host)
        
        # Call Lamport verification kernel
        verify_lamport_sig = _get_function("verify_lamport_sig")
        verify_lamport_sig(
            lamport_sig_device, lamport_pk_device, msg_hash_device, is_valid_device,
            block=(256, 1, 1), grid=(1, 1)
        )
        
        # Read back validity flag
        cuda.memcpy_dtoh(is_valid_host, is_valid_device)
        if is_valid_host[0] == 0:
            return False
            
        # Walk up tree to root candidate on GPU
        path_device = cuda.mem_alloc(len(path))
        cuda.memcpy_htod(path_device, path)
        
        candidate_root_host = np.zeros(32, dtype=np.uint8)
        candidate_root_device = cuda.mem_alloc(32)
        
        lms_verify_tree = _get_function("lms_verify_tree")
        lms_verify_tree(
            lamport_pk_device, path_device, np.int32(q), np.int32(h), candidate_root_device,
            block=(1, 1, 1), grid=(1, 1)
        )
        
        cuda.memcpy_dtoh(candidate_root_host, candidate_root_device)
        candidate_root_bytes = candidate_root_host.tobytes()
        
        return candidate_root_bytes == expected_root


class SimpleSPHINCSGPUSecretKey:
    def __init__(self, seed: bytes, height: int):
        self.seed = seed
        self.height = height
        self.num_leaves = 1 << height


class SimpleSPHINCSGPUBackend:
    def __init__(self, height: int = 4):
        self.height = height
        self.scheme_name = f"sphincs-simple-h{height}"

    def name(self) -> str:
        return "gpu-pycuda"

    def keygen(self) -> Tuple[bytes, SimpleSPHINCSGPUSecretKey]:
        num_leaves = 1 << self.height
        seed = os.urandom(32)
        
        # Allocate GPU memory
        seed_device = cuda.mem_alloc(32)
        cuda.memcpy_htod(seed_device, seed)
        
        public_keys_bytes = num_leaves * 16384
        public_keys_device = cuda.mem_alloc(public_keys_bytes)
        
        tree_bytes = (1 << (self.height + 1)) * 32
        lms_tree_device = cuda.mem_alloc(tree_bytes)
        
        # Initialize tree memory to zero
        zero_tree = np.zeros(tree_bytes, dtype=np.uint8)
        cuda.memcpy_htod(lms_tree_device, zero_tree)
        
        # Generate leaf OTS public keys
        generate_lms_leaf_pks = _get_function("generate_lms_leaf_pks")
        generate_lms_leaf_pks(
            seed_device, public_keys_device,
            block=(512, 1, 1), grid=(num_leaves, 1)
        )
        
        # Hash leaves onto Merkle tree
        hash_lms_leaves = _get_function("hash_lms_leaves")
        threads_per_block = 256
        grid_size = int((num_leaves + threads_per_block - 1) / threads_per_block)
        hash_lms_leaves(
            public_keys_device, lms_tree_device, np.int32(num_leaves),
            block=(threads_per_block, 1, 1), grid=(grid_size, 1)
        )
        
        # Construct tree levels
        build_lms_tree_level = _get_function("build_lms_tree_level")
        for level in range(self.height - 1, -1, -1):
            num_nodes = 1 << level
            level_nodes_start = 1 << level
            
            block_size = min(num_nodes, 256)
            grid_size = int((num_nodes + block_size - 1) / block_size)
            
            build_lms_tree_level(
                lms_tree_device, np.int32(level_nodes_start), np.int32(num_nodes),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            
        # Copy tree root
        root_host = np.zeros(32, dtype=np.uint8)
        cuda.memcpy_dtoh(root_host, int(lms_tree_device) + 32)
        root = root_host.tobytes()
        
        pk = self.height.to_bytes(4, 'big') + root
        sk = SimpleSPHINCSGPUSecretKey(seed, self.height)
        return pk, sk

    def sign(self, sk: SimpleSPHINCSGPUSecretKey, message: bytes) -> bytes:
        # Rebuild tree on GPU to get the path (stateless operation)
        num_leaves = 1 << sk.height
        
        # Allocate GPU memory
        seed_device = cuda.mem_alloc(32)
        cuda.memcpy_htod(seed_device, sk.seed)
        
        public_keys_bytes = num_leaves * 16384
        public_keys_device = cuda.mem_alloc(public_keys_bytes)
        
        tree_bytes = (1 << (sk.height + 1)) * 32
        lms_tree_device = cuda.mem_alloc(tree_bytes)
        
        # Initialize tree memory to zero
        zero_tree = np.zeros(tree_bytes, dtype=np.uint8)
        cuda.memcpy_htod(lms_tree_device, zero_tree)
        
        # Generate leaf OTS public keys
        generate_lms_leaf_pks = _get_function("generate_lms_leaf_pks")
        generate_lms_leaf_pks(
            seed_device, public_keys_device,
            block=(512, 1, 1), grid=(num_leaves, 1)
        )
        
        # Hash leaves onto Merkle tree
        hash_lms_leaves = _get_function("hash_lms_leaves")
        threads_per_block = 256
        grid_size = int((num_leaves + threads_per_block - 1) / threads_per_block)
        hash_lms_leaves(
            public_keys_device, lms_tree_device, np.int32(num_leaves),
            block=(threads_per_block, 1, 1), grid=(grid_size, 1)
        )
        
        # Construct tree levels
        build_lms_tree_level = _get_function("build_lms_tree_level")
        for level in range(sk.height - 1, -1, -1):
            num_nodes = 1 << level
            level_nodes_start = 1 << level
            
            block_size = min(num_nodes, 256)
            grid_size = int((num_nodes + block_size - 1) / block_size)
            
            build_lms_tree_level(
                lms_tree_device, np.int32(level_nodes_start), np.int32(num_nodes),
                block=(block_size, 1, 1), grid=(grid_size, 1)
            )
            
        # Read back tree and public keys
        tree_host = np.zeros(tree_bytes, dtype=np.uint8)
        cuda.memcpy_dtoh(tree_host, lms_tree_device)
        
        public_keys_host = np.zeros(public_keys_bytes, dtype=np.uint8)
        cuda.memcpy_dtoh(public_keys_host, public_keys_device)
        
        # Derive leaf index
        h_idx = hashlib.sha256(b"SPHINCS_LEAF_INDEX" + sk.seed + message).digest()
        leaf_q = int.from_bytes(h_idx[:8], "big") % num_leaves
        
        # Generate private key for leaf_q on host
        sk_leaves = []
        for i in range(512):
            inp = sk.seed + leaf_q.to_bytes(4, 'big') + i.to_bytes(2, 'big')
            sk_leaves.append(hashlib.sha256(inp).digest())
            
        # Hash message and build signature
        msg_hash = hashlib.sha256(message).digest()
        msg_bits = np.unpackbits(np.frombuffer(msg_hash, dtype=np.uint8))
        
        sig_components = []
        for i, bit in enumerate(msg_bits):
            sk_idx = i * 2 + int(bit)
            sig_components.append(sk_leaves[sk_idx])
        signature_bytes = b"".join(sig_components)
        
        # Retrieve Lamport Public Key for leaf_q
        offset = leaf_q * 16384
        public_key_q_bytes = public_keys_host[offset : offset + 16384].tobytes()
        
        # Construct authentication path
        path = []
        node_num = (1 << sk.height) + leaf_q
        while node_num > 1:
            sibling = (node_num - 1) if (node_num % 2 == 1) else (node_num + 1)
            sibling_hash = tree_host[sibling * 32 : (sibling + 1) * 32]
            path.append(sibling_hash)
            node_num //= 2
            
        path_bytes = b"".join(path)
        
        sig = leaf_q.to_bytes(4, 'big') + signature_bytes + public_key_q_bytes + path_bytes
        return sig

    def verify(self, pk: bytes, message: bytes, signature: bytes) -> bool:
        lms_backend = LMSGPUBackend(height=self.height)
        return lms_backend.verify(pk, message, signature)
