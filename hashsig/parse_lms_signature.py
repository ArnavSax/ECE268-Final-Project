def parse_lms_signature(
    signature: bytes,
    height: int
):
    lamport_sig_size = 8192
    lamport_pk_size = 16384

    q = int.from_bytes(signature[:4], "big")

    sig_start = 4
    sig_end = sig_start + lamport_sig_size

    pk_start = sig_end
    pk_end = pk_start + lamport_pk_size

    lamport_sig = signature[sig_start:sig_end]
    lamport_pk = signature[pk_start:pk_end]
    path = signature[pk_end:]

    if len(path) != height * 32:
        raise ValueError("Invalid LMS authentication path")

    return q, lamport_sig, lamport_pk, path
