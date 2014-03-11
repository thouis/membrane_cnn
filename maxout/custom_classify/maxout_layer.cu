#include <float.h>

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
                    if (current_max == -FLT_MAX)
                        current_max = conv_sum;
                }
            }
        }
        OUTPUT(batch_idx, output_i, output_j, output_channel) = current_max;
    }
}



__global__ void softmax_layer(float* input, float* filters, float* bias, float* output,
                              int batches,
                              int input_channels, int input_width, int input_height,
                              int output_channels, int output_width, int output_height,
                              int filter_size)
{
    int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
    int oi = blockIdx.y * blockDim.y + threadIdx.y;
    int oj = blockIdx.z * blockDim.z + threadIdx.z;

#undef INPUT
#undef FILTERS
#undef OUTPUT
#define INPUT(_bi, _i, _j, _ic) input[STRIDED4((_bi), (_i), (_j), (_ic), input_height, input_width, input_channels)]
#define FILTERS(_i, _j, _input_c, _output_c) \
    filters[STRIDED4((_i), (_j), (_input_c), (_output_c), filter_size, input_channels, output_channels)]
#define OUTPUT(_bi, _i, _j, _oc) output[STRIDED4((_bi), (_i), (_j), (_oc), output_height, output_width, output_channels)]


    if (batch_index < batches && oi < output_width && oj < output_height)
    {
        float current_max;

        for(int output_channel = 0; output_channel < output_channels; output_channel++)
        {
            float dot_product = 0;

            // Calculate dot product for output pixel oi, oj
            for (int fi = 0; fi < filter_size; ++fi)
            {
                for (int fj = 0; fj < filter_size; ++fj)
                {
                    for(int input_c = 0; input_c < input_channels; input_c++)
                    {
                        float in_pix = INPUT(batch_index, oi + fi, oj + fj, input_c);
                        float filt_pix = FILTERS(fi, fj, input_c, output_channel);
                        dot_product += in_pix * filt_pix;
                    }
                }
            }

            dot_product += bias[output_channel];

            if ((output_channel == 0) || (dot_product > current_max))
            {
                current_max = dot_product;
            }

            OUTPUT(batch_index, oi, oj, output_channel) = dot_product;
        }

        // Softmax

        float esum = 0;

        for(int output_channel = 0; output_channel < output_channels; ++output_channel )
        {
            float softout = OUTPUT(batch_index, oi, oj, output_channel);
            softout = __expf(softout - current_max);
            esum += softout;
            OUTPUT(batch_index, oi, oj, output_channel) = softout;
        }

        for(int output_channel = 0; output_channel < output_channels; ++output_channel )
        {
            OUTPUT(batch_index, oi, oj, output_channel) /= esum;
        }
    }
}
