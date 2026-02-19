import cupy as cp
import numpy as np
import time

try:
    # Just a simple check
    print(f"Active GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    # Test a smaller matrix first to ensure the 'bridge' works
    size = 2000
    a = cp.random.rand(size, size)
    b = cp.random.rand(size, size)

    start = time.time()
    c = cp.dot(a, b)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
    print(f"Matrix Test Passed in {time.time() - start:.4f}s")

except Exception as e:
    print(f"Still hitting a snag: {e}")