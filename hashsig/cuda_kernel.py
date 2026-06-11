# cudaKernel.py

cuda_code = """
#include <stdint.h>

// SHA-256 logical functions
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

// SHA-256 Constants
__constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Fixed-length SHA-256 helper for block compression
__device__ void sha256_compress(uint32_t state[8], const uint32_t w[64]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for(int i = 0; i < 64; i++) {
        uint32_t S1 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25);
        uint32_t ch = CH(e, f, g);
        uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        
        uint32_t S0 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22);
        uint32_t maj = MAJ(a, b, c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Fixed-length SHA-256 for exactly 32-byte inputs
__device__ void sha256_hash_block(const uint8_t *input, uint8_t *output) {
    uint32_t w[64];
    
    // 1. Initial Hash Values (H0 to H7)
    uint32_t state[8] = { 
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 
    };

    // 2. Prepare the Message Schedule (W)
    for(int i = 0; i < 8; i++) {
        w[i] = (input[i*4] << 24) | (input[i*4+1] << 16) | (input[i*4+2] << 8) | (input[i*4+3]);
    }
    
    // W[8] is the 0x80 padding bit
    w[8] = 0x80000000;
    
    // W[9..14] are zero padding
    for(int i = 9; i < 15; i++) {
        w[i] = 0;
    }
    
    // W[15] is the length of the input in bits (32 bytes * 8 = 256 bits)
    w[15] = 256;

    // 3. Extend the 16 words into 64 words
    for(int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR(w[i-15], 7) ^ ROTR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR(w[i-2], 17) ^ ROTR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    // 4. Compress
    sha256_compress(state, w);

    // 5. Write final state to output array (big-endian format)
    for(int i = 0; i < 8; i++) {
        output[i*4]   = (state[i] >> 24) & 0xFF;
        output[i*4+1] = (state[i] >> 16) & 0xFF;
        output[i*4+2] = (state[i] >> 8)  & 0xFF;
        output[i*4+3] = (state[i])       & 0xFF;
    }
}

// Fixed-length SHA-256 for exactly 64-byte inputs (concatenation of two sibling nodes)
__device__ void sha256_hash_64bytes(const uint8_t *input, uint8_t *output) {
    uint32_t w[64];
    uint32_t state[8] = { 
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 
    };

    // BLOCK 1 (first 64 bytes of message)
    for(int i = 0; i < 16; i++) {
        w[i] = (input[i*4] << 24) | (input[i*4+1] << 16) | (input[i*4+2] << 8) | (input[i*4+3]);
    }
    for(int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR(w[i-15], 7) ^ ROTR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR(w[i-2], 17) ^ ROTR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    sha256_compress(state, w);

    // BLOCK 2 (padding)
    w[0] = 0x80000000;
    for(int i = 1; i < 15; i++) {
        w[i] = 0;
    }
    w[15] = 512; // 64 bytes * 8 bits/byte = 512 bits
    for(int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR(w[i-15], 7) ^ ROTR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR(w[i-2], 17) ^ ROTR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    sha256_compress(state, w);

    // Write final state to output
    for(int i = 0; i < 8; i++) {
        output[i*4]   = (state[i] >> 24) & 0xFF;
        output[i*4+1] = (state[i] >> 16) & 0xFF;
        output[i*4+2] = (state[i] >> 8)  & 0xFF;
        output[i*4+3] = (state[i])       & 0xFF;
    }
}

// Fixed-length SHA-256 for exactly 16384-byte inputs (Lamport OTS public key)
__device__ void sha256_hash_16384bytes(const uint8_t *input, uint8_t *output) {
    uint32_t w[64];
    uint32_t state[8] = { 
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 
    };

    // 256 blocks of 64 bytes
    for (int b = 0; b < 256; b++) {
        const uint8_t *block_ptr = input + (b * 64);
        for(int i = 0; i < 16; i++) {
            w[i] = (block_ptr[i*4] << 24) | (block_ptr[i*4+1] << 16) | (block_ptr[i*4+2] << 8) | (block_ptr[i*4+3]);
        }
        for(int i = 16; i < 64; i++) {
            uint32_t s0 = ROTR(w[i-15], 7) ^ ROTR(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = ROTR(w[i-2], 17) ^ ROTR(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        sha256_compress(state, w);
    }

    // BLOCK 257 (padding)
    w[0] = 0x80000000;
    for(int i = 1; i < 15; i++) {
        w[i] = 0;
    }
    w[15] = 131072; // 16384 bytes * 8 bits/byte = 131072 bits
    for(int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR(w[i-15], 7) ^ ROTR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR(w[i-2], 17) ^ ROTR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    sha256_compress(state, w);

    // Write final state to output
    for(int i = 0; i < 8; i++) {
        output[i*4]   = (state[i] >> 24) & 0xFF;
        output[i*4+1] = (state[i] >> 16) & 0xFF;
        output[i*4+2] = (state[i] >> 8)  & 0xFF;
        output[i*4+3] = (state[i])       & 0xFF;
    }
}

// Each thread verifies ONE of the 256 signature components in parallel
__global__ void verify_lamport_sig(uint8_t *signature, uint8_t *public_key, uint8_t *msg_hash, int *is_valid) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_id < 256) {
        // A. Extract the specific bit of the message hash this thread represents
        int byte_idx = thread_id / 8;
        int bit_idx = 7 - (thread_id % 8); // Extract MSB first
        uint8_t bit = (msg_hash[byte_idx] >> bit_idx) & 1;
        
        // B. Calculate the expected index in the Public Key
        // Each bit position has 2 potential keys. We pick the 0 or 1 based on the bit.
        int pk_idx = (thread_id * 2) + bit;
        
        // C. Hash the provided signature component
        uint8_t *my_sig_component = signature + (thread_id * 32);
        uint8_t computed_hash[32];
        sha256_hash_block(my_sig_component, computed_hash);
        
        // D. Compare the computed hash with the actual public key component
        uint8_t *expected_pk_component = public_key + (pk_idx * 32);
        
        bool match = true;
        for(int i = 0; i < 32; i++) {
            if(computed_hash[i] != expected_pk_component[i]) {
                match = false;
                break;
            }
        }
        
        // E. If ANY thread finds a mismatch, invalidate the whole signature atomically
        if (!match) {
            atomicExch(is_valid, 0);
        }
    }
}

__global__ void generate_lamport_pk(uint8_t *private_key, uint8_t *public_key) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < 512) {
        uint8_t *my_sk_component = private_key + (thread_id * 32);
        uint8_t *my_pk_component = public_key + (thread_id * 32);
        sha256_hash_block(my_sk_component, my_pk_component);
    }
}

__device__ void generate_sk_component(const uint8_t *seed, uint32_t q, uint16_t i, uint8_t *out_sk) {
    uint32_t w[64];
    uint32_t state[8] = { 
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 
    };
    for(int j = 0; j < 8; j++) {
        w[j] = (seed[j*4] << 24) | (seed[j*4+1] << 16) | (seed[j*4+2] << 8) | (seed[j*4+3]);
    }
    w[8] = q;
    w[9] = (i << 16) | 0x8000;
    for(int j = 10; j < 15; j++) { w[j] = 0; }
    w[15] = 38 * 8; // 38 bytes * 8 = 304 bits

    for(int j = 16; j < 64; j++) {
        uint32_t s0 = ROTR(w[j-15], 7) ^ ROTR(w[j-15], 18) ^ (w[j-15] >> 3);
        uint32_t s1 = ROTR(w[j-2], 17) ^ ROTR(w[j-2], 19) ^ (w[j-2] >> 10);
        w[j] = w[j-16] + s0 + w[j-7] + s1;
    }
    sha256_compress(state, w);
    for(int j = 0; j < 8; j++) {
        out_sk[j*4]   = (state[j] >> 24) & 0xFF;
        out_sk[j*4+1] = (state[j] >> 16) & 0xFF;
        out_sk[j*4+2] = (state[j] >> 8)  & 0xFF;
        out_sk[j*4+3] = (state[j])       & 0xFF;
    }
}

__global__ void generate_lms_leaf_pks(const uint8_t *seed, uint8_t *public_keys) {
    int q = blockIdx.x;
    int i = threadIdx.x;
    if (i < 512) {
        uint8_t my_sk[32];
        generate_sk_component(seed, q, i, my_sk);
        
        uint8_t *my_pk_component = public_keys + (q * 16384) + (i * 32);
        sha256_hash_block(my_sk, my_pk_component);
    }
}

__global__ void hash_lms_leaves(uint8_t *public_keys, uint8_t *lms_tree, int num_leaves) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < num_leaves) {
        uint8_t *my_pk = public_keys + (q * 16384);
        uint8_t *my_leaf_node = lms_tree + ((num_leaves + q) * 32);
        sha256_hash_16384bytes(my_pk, my_leaf_node);
    }
}

__global__ void build_lms_tree_level(uint8_t *lms_tree, int level_nodes_start, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        int parent_node_num = level_nodes_start + idx;
        int left_child = 2 * parent_node_num;
        int right_child = 2 * parent_node_num + 1;
        
        uint8_t input_nodes[64];
        for(int j = 0; j < 32; j++) {
            input_nodes[j] = lms_tree[left_child * 32 + j];
            input_nodes[32 + j] = lms_tree[right_child * 32 + j];
        }
        
        uint8_t *my_parent_node = lms_tree + (parent_node_num * 32);
        sha256_hash_64bytes(input_nodes, my_parent_node);
    }
}

__global__ void lms_verify_tree(uint8_t *public_key, uint8_t *path, int q, int h, uint8_t *candidate_root) {
    uint8_t temp[32];
    sha256_hash_16384bytes(public_key, temp);
    
    int node_num = (1 << h) + q;
    for (int i = 0; i < h; i++) {
        uint8_t next_input[64];
        uint8_t *sibling = path + (i * 32);
        
        if (node_num % 2 == 1) {
            // Sibling is left, temp is right
            for(int j = 0; j < 32; j++) {
                next_input[j] = sibling[j];
                next_input[32 + j] = temp[j];
            }
        } else {
            // Temp is left, sibling is right
            for(int j = 0; j < 32; j++) {
                next_input[j] = temp[j];
                next_input[32 + j] = sibling[j];
            }
        }
        sha256_hash_64bytes(next_input, temp);
        node_num /= 2;
    }
    
    for(int j = 0; j < 32; j++) {
        candidate_root[j] = temp[j];
    }
}

__device__ void generate_fors_sk_component(const uint8_t *seed, uint32_t j, uint32_t i, uint8_t *out_sk) {
    uint32_t w[64];
    uint32_t state[8] = { 
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 
    };
    for(int m = 0; m < 8; m++) {
        w[m] = (seed[m*4] << 24) | (seed[m*4+1] << 16) | (seed[m*4+2] << 8) | (seed[m*4+3]);
    }
    w[8] = j;
    w[9] = i;
    w[10] = 0x80000000;
    for(int m = 11; m < 15; m++) { w[m] = 0; }
    w[15] = 40 * 8; // 40 bytes * 8 = 320 bits

    for(int m = 16; m < 64; m++) {
        uint32_t s0 = ROTR(w[m-15], 7) ^ ROTR(w[m-15], 18) ^ (w[m-15] >> 3);
        uint32_t s1 = ROTR(w[m-2], 17) ^ ROTR(w[m-2], 19) ^ (w[m-2] >> 10);
        w[m] = w[m-16] + s0 + w[m-7] + s1;
    }
    sha256_compress(state, w);
    for(int m = 0; m < 8; m++) {
        out_sk[m*4]   = (state[m] >> 24) & 0xFF;
        out_sk[m*4+1] = (state[m] >> 16) & 0xFF;
        out_sk[m*4+2] = (state[m] >> 8)  & 0xFF;
        out_sk[m*4+3] = (state[m])       & 0xFF;
    }
}

__global__ void generate_fors_leaves(const uint8_t *seed, uint8_t *fors_tree, int a, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx goes from 0 to k * 2^a - 1
    int num_leaves_per_tree = 1 << a;
    if (idx < k * num_leaves_per_tree) {
        int j = idx / num_leaves_per_tree; // Tree index
        int i = idx % num_leaves_per_tree; // Leaf index within tree
        
        uint8_t sk[32];
        generate_fors_sk_component(seed, j, i, sk);
        
        // Write directly to leaf node in the fors_tree array
        int tree_size_nodes = 1 << (a + 1);
        int tree_offset = j * tree_size_nodes * 32;
        uint8_t *my_leaf_node = fors_tree + tree_offset + (num_leaves_per_tree + i) * 32;
        sha256_hash_block(sk, my_leaf_node);
    }
}

__global__ void build_fors_tree_level(uint8_t *fors_tree, int a, int level_nodes_start, int level_nodes, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k * level_nodes) {
        int tree_idx = idx / level_nodes;
        int node_idx_in_tree = level_nodes_start + (idx % level_nodes);
        
        int left_child = 2 * node_idx_in_tree;
        int right_child = 2 * node_idx_in_tree + 1;
        
        int tree_size_nodes = 1 << (a + 1);
        int tree_offset = tree_idx * tree_size_nodes * 32;
        
        uint8_t input_nodes[64];
        for(int j = 0; j < 32; j++) {
            input_nodes[j] = fors_tree[tree_offset + left_child * 32 + j];
            input_nodes[32 + j] = fors_tree[tree_offset + right_child * 32 + j];
        }
        
        uint8_t *my_parent_node = fors_tree + tree_offset + node_idx_in_tree * 32;
        sha256_hash_64bytes(input_nodes, my_parent_node);
    }
}

__device__ void sha256_hash_320bytes(const uint8_t *input, uint8_t *output) {
    uint32_t w[64];
    uint32_t state[8] = { 
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 
    };
    for (int b = 0; b < 5; b++) {
        const uint8_t *block_ptr = input + (b * 64);
        for(int i = 0; i < 16; i++) {
            w[i] = (block_ptr[i*4] << 24) | (block_ptr[i*4+1] << 16) | (block_ptr[i*4+2] << 8) | (block_ptr[i*4+3]);
        }
        for(int i = 16; i < 64; i++) {
            uint32_t s0 = ROTR(w[i-15], 7) ^ ROTR(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = ROTR(w[i-2], 17) ^ ROTR(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        sha256_compress(state, w);
    }
    w[0] = 0x80000000;
    for(int i = 1; i < 15; i++) { w[i] = 0; }
    w[15] = 2560; // 320 bytes * 8 = 2560 bits
    for(int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR(w[i-15], 7) ^ ROTR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROTR(w[i-2], 17) ^ ROTR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    sha256_compress(state, w);

    for(int i = 0; i < 8; i++) {
        output[i*4]   = (state[i] >> 24) & 0xFF;
        output[i*4+1] = (state[i] >> 16) & 0xFF;
        output[i*4+2] = (state[i] >> 8)  & 0xFF;
        output[i*4+3] = (state[i])       & 0xFF;
    }
}

__global__ void fors_verify_roots(const uint8_t *revealed_sks, const uint8_t *paths, const int *indices, int a, int k, uint8_t *computed_roots) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < k) {
        uint8_t temp[32];
        const uint8_t *my_sk = revealed_sks + (j * 32);
        sha256_hash_block(my_sk, temp);
        
        int leaf_idx = indices[j];
        int node_num = (1 << a) + leaf_idx;
        
        for (int i = 0; i < a; i++) {
            uint8_t next_input[64];
            const uint8_t *sibling = paths + (j * a * 32) + (i * 32);
            
            if (node_num % 2 == 1) {
                for(int m = 0; m < 32; m++) {
                    next_input[m] = sibling[m];
                    next_input[32 + m] = temp[m];
                }
            } else {
                for(int m = 0; m < 32; m++) {
                    next_input[m] = temp[m];
                    next_input[32 + m] = sibling[m];
                }
            }
            sha256_hash_64bytes(next_input, temp);
            node_num /= 2;
        }
        
        uint8_t *my_root = computed_roots + (j * 32);
        for(int m = 0; m < 32; m++) {
            my_root[m] = temp[m];
        }
    }
}

__global__ void fors_compute_pk(const uint8_t *computed_roots, uint8_t *pk) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sha256_hash_320bytes(computed_roots, pk);
    }
}
"""
