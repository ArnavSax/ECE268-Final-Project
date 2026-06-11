from __future__ import annotations

from hashsig.lamport_cpu import LamportOTS
from hashsig.lms_cpu import LMS
from hashsig.sphincs_simple_cpu import SimpleSPHINCS


def test_lamport():
    msg = b"hello lamport"
    bad = b"hello lamport?"
    pk, sk = LamportOTS.keygen()
    sig = LamportOTS.sign(sk, msg)
    assert LamportOTS.verify(pk, msg, sig)
    assert not LamportOTS.verify(pk, bad, sig)


def test_lms():
    msg = b"hello lms"
    bad = b"hello lms?"
    pk, sk = LMS.keygen(height=3)
    sig = LMS.sign(sk, msg)
    assert LMS.verify(pk, msg, sig)
    assert not LMS.verify(pk, bad, sig)
    sig2 = LMS.sign(sk, b"second message")
    assert sig2.leaf_index == sig.leaf_index + 1


def test_sphincs_simple():
    msg = b"hello simple sphincs"
    bad = b"hello simple sphincs?"
    pk, sk = SimpleSPHINCS.keygen(height=3)
    sig = SimpleSPHINCS.sign(sk, msg)
    assert SimpleSPHINCS.verify(pk, msg, sig)
    assert not SimpleSPHINCS.verify(pk, bad, sig)


def test_lamport_gpu():
    from hashsig.gpu_backend import LamportGPUBackend
    msg = b"hello lamport gpu"
    bad = b"hello lamport gpu?"
    backend = LamportGPUBackend()
    pk, sk = backend.keygen()
    sig = backend.sign(sk, msg)
    assert backend.verify(pk, msg, sig)
    assert not backend.verify(pk, bad, sig)


def test_lms_gpu():
    from hashsig.gpu_backend import LMSGPUBackend
    msg = b"hello lms gpu"
    bad = b"hello lms gpu?"
    backend = LMSGPUBackend(height=3)
    pk, sk = backend.keygen()
    sig = backend.sign(sk, msg)
    assert backend.verify(pk, msg, sig)
    assert not backend.verify(pk, bad, sig)
    assert sk.q == 1
    sig2 = backend.sign(sk, b"second message")
    assert sk.q == 2


def test_sphincs_simple_gpu():
    from hashsig.gpu_backend import SimpleSPHINCSGPUBackend
    msg = b"hello simple sphincs gpu"
    bad = b"hello simple sphincs gpu?"
    backend = SimpleSPHINCSGPUBackend(height=3)
    pk, sk = backend.keygen()
    sig = backend.sign(sk, msg)
    assert backend.verify(pk, msg, sig)
    assert not backend.verify(pk, bad, sig)


def main():
    test_lamport()
    print("Lamport correctness: PASS")
    test_lms()
    print("LMS correctness: PASS")
    test_sphincs_simple()
    print("Simplified SPHINCS correctness: PASS")

    try:
        import pycuda
        test_lamport_gpu()
        print("Lamport GPU correctness: PASS")
        test_lms_gpu()
        print("LMS GPU correctness: PASS")
        test_sphincs_simple_gpu()
        print("Simplified SPHINCS GPU correctness: PASS")
    except ImportError:
        print("PyCUDA not available, skipping GPU tests")


if __name__ == "__main__":
    main()
