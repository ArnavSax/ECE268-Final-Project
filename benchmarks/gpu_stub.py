"""
GPU backend integration stub.

"""


class LamportGPUBackendStub:
    scheme_name = "lamport"

    def name(self) -> str:
        return "gpu-pycuda-stub"

    def keygen(self):
        raise NotImplementedError("Plug PyCUDA Lamport keygen here")

    def sign(self, sk, message: bytes):
        raise NotImplementedError("Plug PyCUDA Lamport sign here")

    def verify(self, pk, message: bytes, signature):
        raise NotImplementedError("Plug PyCUDA Lamport verify here")


"""
GPU timing pattern:

import pycuda.driver as cuda

start = cuda.Event()
end = cuda.Event()

start.record()
# launch kernel
end.record()
end.synchronize()

elapsed_ms = start.time_till(end)

For wall-clock timing from the benchmark harness, the GPU implementation must synchronize:

cuda.Context.synchronize()
"""
