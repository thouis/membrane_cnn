#include <float.h>
#include <stdio.h>

/* NB: these macros assume their arguments have already been parenthesized */
#define STRIDED3(i, j, k, sj, sk) (k + sk * (j + sj * i))
#define STRIDED4(i, j, k, l, sj, sk, sl) (l + sl * STRIDED3(i, j, k, sj, sk))
#define STRIDED5(i, j, k, l, m, sj, sk, sl, sm) (m + sm * STRIDED4(i, j, k, l, sj, sk, sl))


template<unsigned int FILTER_SIZE, unsigned int MAXOUT_SIZE, unsigned int MAXPOOL_SIZE>
__device__ void dev_maxout_layer(float* const __restrict__ input,
                                 float* const __restrict__ filters, 
                                 float* const __restrict__ bias,
                                 float* output,
                                 int batches,
                                 int input_channels, int input_size,
                                 int output_channels, int output_size)
{
    
    int output_i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int output_j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int output_channel_batch = (blockIdx.z * blockDim.z) + threadIdx.z;

    int batch_idx = output_channel_batch / output_channels;
    int output_channel = output_channel_batch - batch_idx * output_channels;


    /* These macros parenthesize their calls to STRIDEDX().  See note above. */
#define INPUT(_bi, _i, _j, _ic) input[STRIDED4((_bi), (_i), (_j), (_ic), input_size, input_size, input_channels)]
#define FILTERS(_i, _j, _mo, _input_c, _output_c) \
    filters[STRIDED5((_i), (_j), (_mo), (_input_c), (_output_c), FILTER_SIZE, MAXOUT_SIZE, input_channels, output_channels)]
#define OUTPUT(_bi, _i, _j, _oc) output[STRIDED4((_bi), (_i), (_j), (_oc), output_size, output_size, output_channels)]
#define BIAS(_i, _j, _oc) bias[STRIDED3((_i), (_j), (_oc), output_size * MAXPOOL_SIZE, output_channels * MAXOUT_SIZE)]

    if ((output_i < output_size) &&
        (output_j < output_size) &&
        (output_channel < output_channels) &&
        (batch_idx < batches)) {

        float current_max = -FLT_MAX;

        // maxpool region
#pragma unroll
        for (int maxpool_i = 0; maxpool_i < MAXPOOL_SIZE; maxpool_i++) {
#pragma unroll
            for (int maxpool_j = 0; maxpool_j < MAXPOOL_SIZE; maxpool_j++) {
                // maxout filters
#pragma unroll
                for (int maxout_index = 0; maxout_index < MAXOUT_SIZE; maxout_index++) {
                    float conv_sum = 0;
                    for (int fi = 0; fi < FILTER_SIZE; ++fi) {
                        for (int fj = 0; fj < FILTER_SIZE; ++fj) {
                            int i = output_i * MAXPOOL_SIZE + maxpool_i;
                            int j = output_j * MAXPOOL_SIZE + maxpool_j;
                            if (i + fi < input_size && j + fj < input_size) {
                                for(int input_c = 0; input_c < input_channels; input_c++) {
                                    // input - fastest variation by channel.
                                    // every thread in a warp reads the same pixel at the same time

                                    float in_pix = INPUT(batch_idx, i + fi, j + fj, input_c);
                                    float filt_pix = FILTERS(fi, fj, maxout_index, input_c, output_channel);
                                    conv_sum += in_pix * filt_pix;
                                }
                            }
                        }
                    }
                    conv_sum += BIAS(output_i * MAXPOOL_SIZE + maxpool_i,
                                     output_j * MAXPOOL_SIZE + maxpool_j,
                                     output_channel * MAXOUT_SIZE + maxout_index);
                    if (conv_sum > current_max)
                        current_max = conv_sum;
                }
            }
        }
        OUTPUT(batch_idx, output_i, output_j, output_channel) = current_max;
    }
}


extern "C" {
__global__ void maxout_layer(float*  input,
                             float* filters, 
                             float* bias,
                             float* output,
                             int batches,
                             int input_channels, int input_size,
                             int output_channels, int output_size,
                             int filter_size, int maxout_size, int maxpool_size)
{
    if (filter_size == 4 && maxout_size == 2 && maxpool_size == 2) {
        dev_maxout_layer<4,2,2>(input, filters, bias, output, batches,
                                input_channels, input_size,
                                output_channels, output_size);
    }
    else if (filter_size == 5 && maxout_size == 4 && maxpool_size == 2) {
        dev_maxout_layer<5,4,2>(input, filters, bias, output, batches,
                                input_channels, input_size,
                                output_channels, output_size);
    }
    else
        printf("NO TEMPLATE for %d %d %d\n", filter_size, maxout_size, maxpool_size);
}

__global__ void softmax_layer(float* input, float* filters, float* bias, float* output,
                              int batches,
                              int num_weights, int num_outputs)
{
    unsigned int tid = threadIdx.x;
    unsigned int batchid = blockIdx.y;
    volatile __shared__ float s[64];
    volatile __shared__ float max;
    volatile __shared__ float sum;

    int input_start = batchid * num_weights;

    max = -FLT_MAX;
    if (tid < 64) {
        for (int output_channel = 0; output_channel < num_outputs; output_channel++) {
            s[tid] = 0.0;
            // load and scale
            for (int base = 0; base < num_weights; base += 64) {
                if (base + tid < num_weights)
                    s[tid] += filters[base + tid + output_channel * num_weights] * input[base + tid + input_start];
            }

            __syncthreads();

            // reduction on single warp
            if (tid < 32) {
                s[tid] += s[tid + 32];
                s[tid] += s[tid + 16];
                s[tid] += s[tid + 8];
                s[tid] += s[tid + 4];
                s[tid] += s[tid + 2];
                s[tid] += s[tid + 1];

                if (tid == 0) {
                    // write back temporary result
                    float tmp = s[tid] + bias[output_channel];
                    output[batchid * num_outputs + output_channel] = tmp;
                    if (tmp > max) max = tmp;
                }
            }
            __syncthreads(); // prevent threads >= 32 from overwriting previous iteration too early.
        }

        // compute softmax
        if (tid < num_outputs) {
            s[tid] = __expf(output[batchid * num_outputs + tid] - max);
            if (tid == 0) {
                sum = 0.0;
                for (int idx = 0; idx < num_outputs; idx++) {
                    sum += s[idx];
                }
            }
            output[batchid * num_outputs + tid] = s[tid] / sum;
        }
    }
}
}
