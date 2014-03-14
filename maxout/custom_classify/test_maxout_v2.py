import os
import os.path
import time

import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray

import numpy as np


pycuda.autoinit.context.set_cache_config(cu.func_cache.PREFER_SHARED)
if os.uname()[0] == 'Darwin':
    nvcc.DEFAULT_NVCC_FLAGS.extend(['-ccbin','/usr/bin/clang', '-Xcompiler', '--stdlib=libstdc++'])
nvcc.DEFAULT_NVCC_FLAGS.append('--ptxas-options=-v')

gpu_maxout_layer_source = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maxout_layer_v2.cu')).read()
kernels = nvcc.SourceModule(gpu_maxout_layer_source, keep=True)
run_layers = kernels.get_function('run_layers')

testim = np.random.uniform(size=(10240, 10240)).astype(np.float32)
filter0 = np.random.uniform(size=(4, 4, 2, 32)).astype(np.float32)
bias0 = np.random.uniform(size=(46, 46, 2, 32)).astype(np.float32)
output0 = np.zeros((23, 23, 32), dtype=np.float32)

testim_gpu = gpuarray.to_gpu(testim)
filter0_gpu = gpuarray.to_gpu(filter0)
bias0_gpu = gpuarray.to_gpu(bias0)
output0_gpu = gpuarray.to_gpu(output0)

st = time.time()

num_repeats = 32
num_pixels=1024
num_streams = 16

streams = [cu.Stream() for idx in range(num_streams)]

block = (256,1,1)
grid = (16,1,1)
for rep in range(num_repeats):
    run_layers(testim_gpu, np.int32(testim.shape[1]),
               np.int32(rep * grid[0]), np.int32(0), np.int32(num_pixels),
               filter0_gpu, bias0_gpu,
               output0_gpu,
               block=block, grid=grid, stream=streams[rep % num_streams])
cu.Context.synchronize()

print (num_repeats * num_pixels * grid[0]) / (time.time() - st), "pixels / sec"

