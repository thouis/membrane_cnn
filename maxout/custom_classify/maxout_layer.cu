__global__ void maxout_layer( float* input, float* filters, float* bias, float* output,
    int batches, int channels, int width, int height,
    int nfilters, int filter_width, int filter_height,
    int output_width, int output_height,
    int maxout_size, int maxpool_size)
{
    int ochannel_index = blockIdx.x * blockDim.x + threadIdx.x;
    int oi = blockIdx.y * blockDim.y + threadIdx.y;
    int oj = blockIdx.z * blockDim.z + threadIdx.z;

    int conv_size = (width - filter_width + 1);
    int conv_size2 = conv_size * conv_size;
    int wh = width * height;
    int input_batchsize = wh * channels;
    int filter_wh = filter_width * filter_height;
    int output_wh = output_width * output_height;
    int output_batchsize = output_wh * (nfilters / maxout_size);

    int start_filter = ochannel_index * maxout_size;
    int end_filter = start_filter + maxout_size - 1;

    if (start_filter < nfilters && oi < output_width && oj < output_height)
    {

        for (int batch_index = 0; batch_index < batches; ++batch_index)
        {

                float current_max;

                // Calculate convolution result for output pixel oi, oj with all filters
                for(int filter_index = start_filter; filter_index <= end_filter; ++filter_index )
                {
                    // Maxpool region
                    for (int i = oi * maxpool_size; i < (oi + 1) * maxpool_size; ++i)
                    {
                        for (int j = oj * maxpool_size; j < (oj + 1) * maxpool_size; ++j)
                        {

                            float conv_sum = 0;

                            // Convolve for all channels
                            for(int c = 0; c < channels; ++c)
                            {
                                for (int fi = 0; fi < filter_width; ++fi)
                                {
                                    for (int fj = 0; fj < filter_height; ++fj)
                                    {
                                        if (i + fi < width && j + fj < height)
                                        {
                                            float in_pix = input[(i + fi) + (j + fj) * width + c * wh + batch_index * input_batchsize];
                                            float filt_pix = filters[fi + fj * filter_width + (filter_index * channels + c) * filter_wh];
                                            conv_sum += in_pix * filt_pix;
                                        }
                                    }
                                }
                            }

                            // Add pixel-wise bias
                            conv_sum += bias[i + j * conv_size + filter_index * conv_size2];

                            // Maxout across channels and maxpool across pixels
                            if (((filter_index % maxout_size == 0) && (i % maxpool_size == 0) && (j % maxpool_size == 0)) ||
                                (conv_sum > current_max))
                            {
                                current_max = conv_sum;
                            }

                        }
                    }

                }
                output[oi + oj * output_width + ochannel_index * output_wh + batch_index * output_batchsize] = current_max;
        }
    }
}
