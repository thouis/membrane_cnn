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
nvcc.DEFAULT_NVCC_FLAGS.extend('--cubin -rdc=true --ptxas-options=-v'.split(' '))

gpu_maxout_layer_source = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maxout_layer_v2.cu')).read()
kernels = nvcc.SourceModule(gpu_maxout_layer_source, keep=True)
load_filters = kernels.get_function('load_filters')
run_layers = kernels.get_function('run_layers')

testim = np.random.uniform(size=(10240, 10240)).astype(np.float32)
filter0 = np.random.uniform(size=(4, 4, 2, 32)).astype(np.float32)
bias0 = np.random.uniform(size=(46, 46, 2, 32)).astype(np.float32)
filter1 = np.random.uniform(size=(4, 4, 32, 32, 2)).astype(np.float32)
bias1 = np.random.uniform(size=(20, 20, 2, 32)).astype(np.float32)


scratch = np.zeros(np.prod(23 * 23 * 32) + (10 * 10 * 32) + (3 * 3 * 32)).astype(np.float32)

testim_gpu = gpuarray.to_gpu(testim)
scratch_gpu = gpuarray.to_gpu(scratch)

load_filters(filter0, bias0, filter1, bias1, block=1, grid=1)
cu.Context.synchronize()

st = time.time()

num_repeats = 1
num_pixels=1024
num_streams = 1

streams = [cu.Stream() for idx in range(num_streams)]

block = (256,1,1)
grid = (40,1,1)
for rep in range(num_repeats):
    run_layers(testim_gpu, np.int32(testim.shape[1]),
               np.int32(rep * grid[0]), np.int32(0), np.int32(num_pixels),
               scratch_gpu,
               block=block, grid=grid, stream=streams[rep % num_streams])
cu.Context.synchronize()

print (num_repeats * num_pixels * grid[0]) / (time.time() - st), "pixels / sec"

