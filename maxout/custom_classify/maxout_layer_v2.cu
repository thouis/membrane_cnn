#include <float.h>
#include <stdio.h>

/* NB: these macros assume their arguments have already been parenthesized */
#define STRIDED3(i, j, k, sj, sk) (k + sk * (j + sj * i))
#define STRIDED4(i, j, k, l, sj, sk, sl) (l + sl * STRIDED3(i, j, k, sj, sk))
#define STRIDED5(i, j, k, l, m, sj, sk, sl, sm) (m + sm * STRIDED4(i, j, k, l, sj, sk, sl))

#define FILTER0(i, j, output_channel, maxout_idx) filter0[STRIDED4((i), (j), (output_channel), (maxout_idx), 4, 32, 2)]
#define BIAS0(i, j, output_channel, lane_outchan) bias0[STRIDED4((i), (j), (output_channel), (lane_outchan), 46, 32, 8)]
#define OUTPUT0(_i, _j, _oc) output0[STRIDED3((_i), (_j), (_oc), 23, 32)]


__global__ void run_layers(const unsigned char * image, int image_width,
                           int start_row, int start_col, int end_col,
                           const unsigned short * __restrict__ filter0,
                           const unsigned short * __restrict__ bias0,
                           float *output0)
{
    // Assume a 256x1x1 block of threads
    int tid = threadIdx.x;
    start_row += blockIdx.x;

    // grouping by sets of 128 or 64 (initially, 128)
    int tid_row = tid / 128;
    int tid_col = tid % 128;

    // Make sure threads that maxpool/maxout together are in the same warp
    int id_outchan = tid / 8;
    int lane_outchan = tid % 8;
    int id_maxout = tid % 2;
    int id_maxpool_row = (tid / 2) % 2;
    int id_maxpool_col = (tid / 4) % 2;
    
    __shared__ unsigned char input[49][128];

    // load filter for this thread's output channel
    unsigned short filter[4][4];
    for (int filter_row = 0; filter_row < 4; filter_row++)
        for (int filter_col = 0; filter_col < 4; filter_col++)
            filter[filter_row][filter_col] = FILTER0(filter_row, filter_col, id_outchan, id_maxout);


    // Preload input array.  Use input_offset to shift the location
    // being processed.  When it reaches 64, shift everything left and
    // load a new block.  We can assume image[] is 256-byte aligned.
    int input_offset = (start_row * image_width - start_col) % 64;
    for (int row = 0; row < 49; row += 2) {
        if ((row + tid_row < 49) && (input_offset + tid_col < 128))
            input[row + tid_row][input_offset + tid_col] = 
                image[(start_row + row + tid_row) * image_width + start_col + tid_col];
                                
    }
    __syncthreads();
    // switch to grouping by sets of 64
    tid_row = tid / 64;
    tid_col = tid % 64;


    for (int classify_col = start_col; classify_col < end_col; classify_col++) {
        if (input_offset == 64) {
            // shift everything left, load a new sets of 64 input,
            // and reset the offset.
            // 
            // We assume the image has enough padding on the right
            // side to allow this.
            __syncthreads(); // wait for everyone to catch up

            for (int row = 0; row < 49; row += 4) {
                if (row + tid_row < 49) {
                    input[row + tid_row][tid_col] = input[row + tid_row][64 + tid_col];
                    input[row + tid_row][64 + tid_col] = \
                        image[(start_row + row + tid_row) * image_width + classify_col + 64 + tid_col];
                }
            }
            input_offset = 0;

            __syncthreads(); // do not proceed until new data is ready
        }
            
        // Each thread computes the output for one voxel for the pre-maxout-pre-maxpool 46x46x64 output array.
        //
        // (input_row / 2) and (output_row / 2) are the location of the eventual output in the 23x23x32 output.
        for (int input_row=id_maxpool_row; input_row < 45; input_row += 2) {
            for (int input_col=id_maxpool_col; input_col < 45; input_col += 2) {
                // Compute the output for this location given this thread's maxpool/maxout/outchan
                float convsum = __half2float(BIAS0(input_row / 2, input_col / 2, id_outchan, lane_outchan));
#pragma unroll
                for (int filter_row = 0; filter_row < 4; filter_row++)
#pragma unroll
                    for (int filter_col = 0; filter_col < 4; filter_col++) {
                        unsigned char tmp = \
                            input[input_row + filter_row][input_col + filter_col + input_offset];
                        convsum += __half2float(filter[filter_row][filter_col]) * ((float) tmp) / 256.0;
                    }
                // Now, we use _shufl_down to find the max in each set
                // of threads with the same outchan.  These are all in
                // the same warp.  See the initialization of id_max*
                // above.
                convsum = fmaxf(convsum, __shfl_down(convsum, 4));
                convsum = fmaxf(convsum, __shfl_down(convsum, 2));
                convsum = fmaxf(convsum, __shfl_down(convsum, 1));
                // only the 0,0,0 maxout/maxpool thread writes output
                if (lane_outchan == 0)
                    OUTPUT0(input_row / 2, input_col / 2, id_outchan) = convsum;
            }
        }
        input_offset++;
        __syncthreads();// because we're going to need to launch more threads here.
    }
}
