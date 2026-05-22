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


def main():
    test_lamport()
    print("Lamport correctness: PASS")
    test_lms()
    print("LMS correctness: PASS")
    test_sphincs_simple()
    print("Simplified SPHINCS correctness: PASS")


if __name__ == "__main__":
    main()
