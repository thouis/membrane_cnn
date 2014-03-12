#include <float.h>
#include <stdio.h>

/* NB: these macros assume their arguments have already been parenthesized */
#define STRIDED3(i, j, k, sj, sk) (k + sk * (j + sj * i))
#define STRIDED4(i, j, k, l, sj, sk, sl) (l + sl * STRIDED3(i, j, k, sj, sk))
#define STRIDED5(i, j, k, l, m, sj, sk, sl, sm) (m + sm * STRIDED4(i, j, k, l, sj, sk, sl))


__global__ void maxout_layer(float* input, float* filters, float* bias, float* output,
                             int batches,
                             int input_channels, int input_width, int input_height,
                             int output_channels, int output_width, int output_height,
                             int filter_size, int maxout_size, int maxpool_size)
{
    int output_i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int output_j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int output_channel_batch = (blockIdx.z * blockDim.z) + threadIdx.z;

    int batch_idx = output_channel_batch / output_channels;
    int output_channel = output_channel_batch - batch_idx * output_channels;


    /* These macros parenthesize their calls to STRIDEDX().  See note above. */
#define INPUT(_bi, _i, _j, _ic) input[STRIDED4((_bi), (_i), (_j), (_ic), input_height, input_width, input_channels)]
#define FILTERS(_i, _j, _mo, _input_c, _output_c) \
    filters[STRIDED5((_i), (_j), (_mo), (_input_c), (_output_c), filter_size, maxout_size, input_channels, output_channels)]
#define OUTPUT(_bi, _i, _j, _oc) output[STRIDED4((_bi), (_i), (_j), (_oc), output_height, output_width, output_channels)]
#define BIAS(_i, _j, _oc) bias[STRIDED3((_i), (_j), (_oc), output_width * maxpool_size, output_channels * maxout_size)]

    if ((output_i < output_height) &&
        (output_j < output_width) &&
        (output_channel < output_channels) &&
        (batch_idx < batches)) {

        float current_max = -FLT_MAX;

        // maxpool region
        int base_input_i = output_i * maxpool_size;
        int base_input_j = output_j * maxpool_size;
        for (int maxpool_i = 0; maxpool_i < maxpool_size; maxpool_i++) {
            for (int maxpool_j = 0; maxpool_j < maxpool_size; maxpool_j++) {
                // maxout filters
                for (int maxout_index = 0; maxout_index < maxout_size; maxout_index++) {
                    float conv_sum = 0;
                    for (int fi = 0; fi < filter_size; ++fi) {
                        for (int fj = 0; fj < filter_size; ++fj) {
                            int i = base_input_i + maxpool_i;
                            int j = base_input_j + maxpool_j;
                            if (i + fi < input_height && j + fj < input_width) {
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
                    conv_sum += BIAS(output_i * maxpool_size + maxpool_i,
                                     output_j * maxpool_size + maxpool_j,
                                     output_channel * maxout_size + maxout_index);
                    if (conv_sum > current_max)
                        current_max = conv_sum;
                }
            }
        }
        OUTPUT(batch_idx, output_i, output_j, output_channel) = current_max;
    }
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
