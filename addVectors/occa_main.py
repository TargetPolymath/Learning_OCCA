from ctypes import *
import occa
import numpy as np

import time

entries = 31

a  = np.arange(entries, dtype=np.float32)
b  = a + 2
ab = np.zeros(entries, dtype=np.float32)

Serial_Info   = {"mode": "Serial"}
OpenMP_Info   = {"mode": "OpenMP"  , "schedule": "compact", "chunk": 10}
OpenCL_Info   = {"mode": "OpenCL"  , "platform_id": 0, "device_id": 0}
CUDA_Info     = {"mode": "CUDA"    , "device_id": 0}
Pthreads_Info = {"mode": "Pthreads", "thread_count": 4, "schedule": "compact", "pinned_cores": [0, 0, 1, 1]}
COI_Info      = {"mode": "COI"     , "device_id": 0}

device = occa.Device(CUDA_Info)

o_a  = device.malloc(a , dtype=np.float32)
o_b  = device.malloc(b , dtype=np.float32)
o_ab = device.malloc(ab, dtype=np.float32)

addVectors = device.build_kernel("addVectors.okl",
                                          "addVectors")
millis = time.time()

iters = 100

# for i in range(iters):
# 	addVectors(np.intc(entries),
#             o_a,
#             o_b, 
#             o_ab)
# 	o_ab.copy_to(ab)

addVectors(np.intc(entries), o_a, o_b, o_ab);
o_ab.copy_to(ab)
print((time.time() - millis)*(1000))

print(ab)


# print(ab)