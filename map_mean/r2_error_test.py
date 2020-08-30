from ctypes import *
import occa
import numpy as np
import math
import time

BLOCK_SIZE = 64

w = 4290
h = 2856
pixels = w*h;
entries = 4*pixels

a = np.random.uniform(low = 0.0, high = 255, size = (entries))
b = a + np.random.uniform(low = -1, high = 1, size = (entries));
straight_e = np.zeros((pixels))
scaled_e = np.zeros((math.ceil(pixels / BLOCK_SIZE)))


t_straight_e = np.zeros((pixels))
t_scaled_e = np.zeros((math.ceil(pixels / BLOCK_SIZE)))

# print(a.reshape((10, 10, 4)))


Serial_Info   = {"mode": "Serial"}
OpenMP_Info   = {"mode": "OpenMP"  , "schedule": "compact", "chunk": 10}
OpenCL_Info   = {"mode": "OpenCL"  , "platform_id": 0, "device_id": 0}
CUDA_Info     = {"mode": "CUDA"    , "device_id": 0}
Pthreads_Info = {"mode": "Pthreads", "thread_count": 4, "schedule": "compact", "pinned_cores": [0, 0, 1, 1]}
COI_Info      = {"mode": "COI"     , "device_id": 0}


device = occa.Device(CUDA_Info)


ar1 = device.malloc(a, dtype=np.float32)
ar2 = device.malloc(b, dtype=np.float32)
o_straight_e = device.malloc(straight_e, dtype = np.float32)
o_scaled_e = device.malloc(scaled_e, dtype = np.float32)


map_r2 = device.build_kernel("map_r2.okl", "map_r2")

millis = time.time()

iters = 10

for i in range(iters):
	# ar1.copy_from(a);
	# ar2.copy_from(b);
	o_straight_e.copy_from(straight_e)
	o_scaled_e.copy_from(scaled_e)

	map_r2(np.intc(pixels), ar1, ar2, o_straight_e, o_scaled_e);

	o_straight_e.copy_to(t_straight_e)
	o_scaled_e.copy_to(t_scaled_e)
	straight_errors = t_straight_e.reshape((w, h))
	# print(straight_errors)
	# print("\n"*2)



	blocksum = t_scaled_e; # [::(BLOCK_SIZE*4)]
	# print(blocksum)
	# print("\n"*2)
	print(np.sum(blocksum)/pixels)

print(((time.time() - millis)*(1000)) / iters)
o_straight_e.copy_to(t_straight_e)
o_scaled_e.copy_to(t_scaled_e)
# map_r2(np.intc(pixels), ar1, ar2);

# ar1.copy_to(a)


np.set_printoptions(suppress=True)



print("Numpy test")
millis = time.time()
for i in range(iters):
	ar = a.reshape((w, h, 4))
	br = b.reshape((w, h, 4))
	test = np.mean(np.mean(np.square(np.subtract(ar[:, :, :3], br[:, :, :3])), -1) * (np.multiply(ar[:, :, -1], br[:, :, -1])/(255*255)))
	print(test)
print(((time.time() - millis)*(1000)) / iters)
# a = a.reshape((10,10, 4))


# print(a)


# print(ab)

device.finish()