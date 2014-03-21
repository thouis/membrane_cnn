# Library for full image cnn operations

import numpy as np
import scipy.ndimage
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from numpy.fft import rfftn
from numpy.fft import irfftn
import mahotas
import time
import h5py
import os
import os.path

import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray

pycuda.autoinit.context.set_cache_config(cu.func_cache.PREFER_L1)

BLOCK_BATCHES = 512
BLOCK_PIXELS = 1

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

if os.uname()[0] == 'Darwin':
    nvcc.DEFAULT_NVCC_FLAGS.extend(['-ccbin','/usr/bin/clang', '-Xcompiler', '--stdlib=libstdc++'])
nvcc.DEFAULT_NVCC_FLAGS.append('--ptxas-options=-v')


gpu_maxout_layer_source = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maxout_layer.cu')).read()

kernels = nvcc.SourceModule(gpu_maxout_layer_source, no_extern_c=True)
gpu_maxout_layer = kernels.get_function('maxout_layer')
gpu_softmax_layer = kernels.get_function('softmax_layer')

class MaxoutMaxpoolLayer(object):
    def __init__(self, nkernels, ninputs, kernel_size, stride_in, maxpool_size, maxout_size, W=None, b=None):
        self.ninputs = ninputs
        self.nkernels = nkernels
        self.kernel_size = kernel_size
        self.maxpool_size = maxpool_size
        self.maxout_size = maxout_size
        self.stride_in = stride_in
        self.stride_out = stride_in
        self.noutputs = nkernels / maxout_size
        # Size of previous convolution operation (for fft result cache)
        self.prev_conv_size = 0
        # Input / output footprint - set once full network has been constructed
        self.input_footprint = 0
        self.output_footprint = 0

        # original dimensions of W: [filter index, input channel, i, j]
        # where filter index = output_channel * maxout_size + maxout_offset
        # rearrange W as [i, j, maxout_offset, input_channel, output_channel]
        W2 = W.transpose((2, 3, 1, 0))
        assert W2.shape ==  (kernel_size, kernel_size, ninputs, nkernels)
        W2 = W2.reshape((kernel_size, kernel_size, ninputs, self.noutputs, maxout_size))
        W2 = W2.transpose((0, 1, 4, 2, 3))
        assert W2.shape ==  (kernel_size, kernel_size, maxout_size, ninputs, self.noutputs)

        # b is (output_channel * maxout_size, output_height * maxpool_size, output_width * maxpool_size)
        # transpose to (h, w, oc)
        b2 = b.transpose((1, 2, 0))

        self.W = gpuarray.to_gpu(W2.copy())
        self.b = gpuarray.to_gpu(b2.copy())

    def apply_layer(self, input_image, nbatches):
        output_size = (nbatches, self.output_footprint, self.output_footprint, self.noutputs, )
        print "   ", input_image.shape, '->', output_size

        block = (2, 4, self.noutputs)
        grid = (((self.output_footprint - 1) / block[0]) + 1,
                ((self.output_footprint - 1) / block[1]) + 1,
                nbatches)

        if not isinstance(input_image, gpuarray.GPUArray):
            input_image = gpuarray.to_gpu(input_image)

        d_maxout_result = gpuarray.zeros(long(np.prod(output_size)), np.float32).reshape(output_size)

        gpu_maxout_layer(input_image, self.W, self.b, d_maxout_result,
                         np.int32(nbatches),
                         np.int32(self.ninputs), np.int32(self.input_footprint),
                         np.int32(self.noutputs), np.int32(self.output_footprint),
                         np.int32(self.kernel_size), np.int32(self.maxout_size), np.int32(self.maxpool_size),
                         block=block, grid=grid)

        print "    MO Layer: Complete."

        return d_maxout_result


class SoftmaxLayer(object):
    def __init__(self, ninputs, noutputs, kernel_size, stride, W, b):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.kernel_size = kernel_size
        self.stride_in = stride
        self.stride_out = stride
        # Input / output footprint - set once full network has been constructed
        self.input_footprint = 0
        self.output_footprint = 0

        self.W = gpuarray.to_gpu(W.T.copy())
        self.b = gpuarray.to_gpu(b)

    def apply_layer(self, input_image, nbatches):

        # Calculate feed-forward result
        output_size = (nbatches, self.noutputs)
        print "   ", input_image.shape, '->', output_size

        block = (64, 1, 1)
        grid = (1, nbatches)

        if not isinstance(input_image, gpuarray.GPUArray):
            input_image = gpuarray.to_gpu(input_image)

        d_softmax_result = gpuarray.zeros(long(np.prod(output_size)), np.float32).reshape(output_size)

        gpu_softmax_layer(input_image, self.W, self.b, d_softmax_result,
                          np.int32(nbatches),
                          np.int32(self.W.shape[1]), np.int32(self.W.shape[0]),
                          block=block, grid=grid)

        print "    SM Layer: Complete."

        return d_softmax_result


class DeepNetwork(object):
    def __init__(self, filename):

        network_h5 = h5py.File(filename, 'r')

        self.nlayers = network_h5['/layers'][...]

        print 'Network has {0} layers.'.format(self.nlayers)

        if '/downsample_factor' in network_h5:
            self.downsample = network_h5['/downsample_factor'][...]
        else:
            self.downsample = 1

        self.best_sigma = 0
        self.best_offset = (0,0)

        all_layers = []
        stride_in = 1

        for layer_i in range(self.nlayers):

            layer_string = '/layer{0}/'.format(layer_i)
            layer_type = network_h5[layer_string + 'type'][...]

            if layer_type == 'MaxoutConvC01B':

                layer_weights = network_h5[layer_string + 'weights'][...]
                layer_bias = network_h5[layer_string + 'bias'][...]
                layer_maxpoolsize = network_h5[layer_string + 'pool_shape'][...][0]
                layer_maxoutsize = network_h5[layer_string + 'num_pieces'][...]

                # Arrange weights as [kernels, inputs, ksize, ksize]
                layer_weights = np.rollaxis(layer_weights, 3, 0)

                new_layer = MaxoutMaxpoolLayer(
                    layer_weights.shape[0], layer_weights.shape[1], layer_weights.shape[2],
                    stride_in, layer_maxpoolsize, layer_maxoutsize, W=layer_weights, b=layer_bias)

            elif layer_type == 'Softmax':

                layer_weights = network_h5[layer_string + 'weights'][...]
                layer_bias = network_h5[layer_string + 'bias'][...]
                layer_ksize = network_h5[layer_string + 'ksize'][...][0]

                new_layer = SoftmaxLayer(
                    layer_weights.shape[0] / (layer_ksize ** 2), layer_weights.shape[1], layer_ksize,
                    stride_in, W=layer_weights, b=layer_bias)

            else:
                raise Exception("Unknown layer type: {0}".format(layer_type))

            all_layers.append(new_layer)

            stride_in = new_layer.stride_out

        # Calculate network footprint and therefore pad size
        footprint = 1
        for layer in range(self.nlayers-1, -1, -1):
            all_layers[layer].output_footprint = footprint
            if layer == self.nlayers - 1:
                footprint = all_layers[layer].kernel_size
            else:
                footprint = footprint * all_layers[layer].maxpool_size - 1 + all_layers[layer].kernel_size
            all_layers[layer].input_footprint = footprint

        self.all_layers = all_layers
        self.pad_by = int(self.downsample * (footprint // 2))


    def apply_net(self, input_image, perform_downsample=False, perform_pad=False, perform_upsample=False, perform_blur=False, perform_offset=False):

        if perform_pad:
            input_image = np.pad(input_image, ((self.pad_by, self.pad_by), (self.pad_by, self.pad_by)), 'symmetric')

        if perform_downsample and self.downsample != 1:
            input_image = np.float32(mahotas.imresize(input_image, 1.0/self.downsample))

        nx = input_image.shape[0] - self.all_layers[0].input_footprint + 1
        ny = input_image.shape[1] - self.all_layers[0].input_footprint + 1
        nbatches = nx * ny

        layer_temp = np.zeros((nbatches, self.all_layers[0].input_footprint, self.all_layers[0].input_footprint, 1), dtype=np.float32)
        print layer_temp.shape

        batchi = 0
        for x in range(nx):
            for y in range(ny):
                #print (x,y)
                layer_temp[batchi, :, :, 0] = input_image[x:(x + self.all_layers[0].input_footprint), y:(y + self.all_layers[0].input_footprint)]
                batchi += 1

        assert batchi == nbatches

        output = np.zeros(nbatches, dtype=np.float32)

        count = 0
        batch_start_time = time.time()
        for block_from in range(0, nbatches, BLOCK_BATCHES):
            count = count + 1

            block_to = min(block_from + BLOCK_BATCHES, layer_temp.shape[0])
            batchsize = block_to - block_from
            print "Block", block_from, block_to, " (of", nbatches, ")"

            block_temp = layer_temp[block_from:block_to,:,:,:]

            for layeri in range(len(self.all_layers)):
                print "running layer", layeri
                start_time = time.clock()
                block_temp = self.all_layers[layeri].apply_layer(block_temp, batchsize)
                end_time = time.clock()
                print('    Layer time = %.3fm' % ((end_time - start_time) / 60.))
            print "    %.3fm used so far, %.3fm total estimated" % ((time.time() - batch_start_time) / 60.0,
                                                                    nbatches * (time.time() - batch_start_time) / (60.0 * block_to))
            print ""

            if isinstance(block_temp, gpuarray.GPUArray):
                block_temp = block_temp.get()

            output[block_from:block_to] = block_temp[:,0]

        output = output.reshape(nx, ny)

        if perform_upsample:
            output = np.float32(mahotas.imresize(output, self.downsample))

        if perform_blur and self.best_sigma != 0:
            output = scipy.ndimage.filters.gaussian_filter(output, self.best_sigma)

        if perform_offset:
            #Translate
            output = np.roll(output, self.best_offset[0], axis=0)
            output = np.roll(output, self.best_offset[1], axis=1)

        # Crop to valid size
        #output = output[self.pad_by:-self.pad_by,self.pad_by:-self.pad_by]
        return output


def done():
    pycuda.autoinit.context.detach()
