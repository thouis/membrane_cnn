#include <float.h>
#include <stdio.h>

/* NB: these macros assume their arguments have already been parenthesized */
#define STRIDED3(i, j, k, sj, sk) (k + sk * (j + sj * i))
#define STRIDED4(i, j, k, l, sj, sk, sl) (l + sl * STRIDED3(i, j, k, sj, sk))
#define STRIDED5(i, j, k, l, m, sj, sk, sl, sm) (m + sm * STRIDED4(i, j, k, l, sj, sk, sl))

#define OUTPUT_SIZE_0 (23 * 23 * 32)
#define OUTPUT_SIZE_1 (10 * 10 * 32)
#define OUTPUT_SIZE_2 (3 * 3 * 32)
#define OUTPUT_SIZE_3 (2)

__device__ const unsigned short * __restrict__ filter0;
__device__ const unsigned short * __restrict__ bias0;
__device__ const unsigned short * __restrict__ filter1;
__device__ const unsigned short * __restrict__ bias1;


__global__
__launch_bounds__(1)
void load_filters(unsigned short *f0, unsigned short *b0, unsigned short *f1, unsigned short *b1)
{
    filter0 = (const unsigned short *) f0;
    bias0 = (const unsigned short *) b0;
    filter1 = (const unsigned short *) f1;
    bias1 = (const unsigned short *) b1;
}

#define FILTER1(i, j, input_channel, output_channel, maxout_idx) __ldg(& filter1[STRIDED5((i), (j), (output_channel), (maxout_idx), (input_channel), 4, 32, 2, 32)])
#define BIAS1(i, j, id_maxout, output_channel) __ldg(& bias1[STRIDED4((i), (j), (output_channel), (id_maxout), 20, 32, 2)])
#define INPUT1(_i, _j, _oc) output0[STRIDED3((_i), (_j), (_oc), 23, 32)]
#define OUTPUT1(_i, _j, _oc) output1[STRIDED3((_i), (_j), (_oc), 10, 32)]
#define MAX_THREADS_PER_BLOCK_1 128


__global__
__launch_bounds__(MAX_THREADS_PER_BLOCK_1)
void layer1(const float * __restrict__ output0, float *output1, int maxout_idx)
{
    int out_chan = blockIdx.x;
    int out_row = blockIdx.y;
    int in_chan = threadIdx.x;
    int maxpool_row = threadIdx.y;
    int maxpool_col = threadIdx.z;

    __shared__ float maxpool_buffer[2][2];

    volatile int in_row = 2 * out_row + maxpool_row;

    for (int out_col = 0; out_col < 10; out_col++) {
        volatile int in_col = 2 * out_col + maxpool_col;

        // initialize to previous maxpool/maxout value
        volatile float max;
        if (maxout_idx == 0)
            max = -FLT_MAX;
        else
            max = OUTPUT1(out_row, out_col, out_chan);

        volatile float convsum = 0.0;
#pragma unroll
        for (int filter_row = 0; filter_row < 4; filter_row++)
#pragma unroll
            for (int filter_col = 0; filter_col < 4; filter_col++) {
                float tmp =                                                 \
                    INPUT1(in_row + filter_row,
                           in_col + filter_col,
                           in_chan);
                convsum += __half2float(FILTER1(filter_row, filter_col, out_chan, maxout_idx, in_chan)) * tmp;
            }
        // Every thread now has part of the convolution.
        // 
        // Use in-warp shuffle to sum them to thread 0
        convsum += __shfl_down(convsum, 16);
        convsum += __shfl_down(convsum, 8);
        convsum += __shfl_down(convsum, 4);
        convsum += __shfl_down(convsum, 2);
        convsum += __shfl_down(convsum, 1);
        convsum += BIAS1(in_row,
                         in_col,
                         out_chan, maxout_idx);
        max = fmaxf(max, convsum);
        if (in_chan == 0)
            maxpool_buffer[maxpool_row][maxpool_col] = max;
        __syncthreads();
        if (in_chan == 0 && maxpool_row == 0 && maxpool_col == 0) {
            max = fmaxf(fmaxf(maxpool_buffer[0][0], maxpool_buffer[0][1]),
                        fmaxf(maxpool_buffer[1][0], maxpool_buffer[1][1]));
            OUTPUT1(out_row, out_col, out_chan) = max;
        }
    }
}

#define FILTER0(i, j, output_channel, maxout_idx) __ldg(& filter0[STRIDED4((i), (j), (output_channel), (maxout_idx), 4, 32, 2)])
#define BIAS0(i, j, output_channel, lane_outchan) __ldg(& bias0[STRIDED4((i), (j), (output_channel), (lane_outchan), 46, 32, 8)])
#define OUTPUT0(_i, _j, _oc) scratch[STRIDED3((_i), (_j), (_oc), 23, 32)]
#define MAX_THREADS_PER_BLOCK_0 256

__global__
__launch_bounds__(MAX_THREADS_PER_BLOCK_0)
void run_layers(const unsigned char *image, int image_width,
                int start_row, int start_col, int end_col,
                float *scratch)
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

    // load filter for this thread's output channels
    unsigned short filter[4][4];
    for (int filter_row = 0; filter_row < 4; filter_row++)
        for (int filter_col = 0; filter_col < 4; filter_col++)
            filter[filter_row][filter_col] = FILTER0(filter_row, filter_col, id_outchan, id_maxout);


    // Preload input array.  Use input_offset to shift the location
    // being processed.  When it reaches 64, shift everything left and
    // load a new block.  We can assume image[] is 256-byte aligned.
    int input_offset = (start_row * image_width + start_col) % 64;
    for (int row = tid_row; row < 49; row += 2) {
        if ((input_offset + tid_col < 128))
            input[row][input_offset + tid_col] = 
                __ldg(& image[(start_row + row) * image_width + start_col + tid_col]);
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

            for (int row = tid_row; row < 49; row += 4) {
                input[row][tid_col] = input[row][64 + tid_col];
                input[row][64 + tid_col] = __ldg(& image[(start_row + row) * image_width + classify_col + 64 + tid_col]);
            }
            input_offset = 0;

            __syncthreads(); // do not proceed until input array is ready
        }
            
        // Each thread computes the output for one voxel for the pre-maxout-pre-maxpool 46x46x64 output array.
        //
        // (input_row / 2) and (output_row / 2) are the location of the eventual output in the 23x23x32 output.
        for (int input_row=id_maxpool_row; input_row < 46; input_row += 2) {
            for (int input_col=id_maxpool_col; input_col < 46; input_col += 2) {
                float convsum = __half2float(BIAS0(input_row, input_col, id_outchan, lane_outchan));
                for (int filter_row = 0; filter_row < 4; filter_row++)
                    for (int filter_col = 0; filter_col < 4; filter_col++) {
                        unsigned char tmp = \
                            input[input_row + filter_row][input_col + filter_col + input_offset];
                        convsum += __half2float(filter[filter_row][filter_col]) * ((float) tmp) / 256.0;
                    }
                // Use _shufl_down to find the max in each set of
                // threads with the same outchan.  These are all in
                // the same warp.  See the initialization of id_max*
                // above.
                convsum = fmaxf(convsum, __shfl_down(convsum, 1));
                convsum = fmaxf(convsum, __shfl_down(convsum, 2));
                convsum = fmaxf(convsum, __shfl_down(convsum, 4));
                // only the 0,0,0 maxout/maxpool thread writes output
                if (lane_outchan == 0)
                    OUTPUT0(input_row / 2, input_col / 2, id_outchan) = convsum;
            }
        }
        input_offset++;

        __syncthreads();
        if (tid == 0) {
            dim3 grid(32, 10, 1), block(32, 2, 2);
            // maxout 0
            layer1<<<grid, block, 32>>>(scratch, scratch + OUTPUT_SIZE_0, 0);
            // maxout 1
            layer1<<<grid, block, 32>>>(scratch, scratch + OUTPUT_SIZE_0, 1);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf( "Error! %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();
        }
        __syncthreads();

    }
}



void allocate(void **ptr, int sz)
{
    if (cudaMalloc(ptr, sz) != cudaSuccess) {
        printf("failure allocating\n");
    }
}

int main(int argc, char *argv[])
{
    unsigned char *testim;
    unsigned short *filter0, *bias0, *filter1, *bias1;
    float *scratch;

    cudaThreadSetCacheConfig(cudaFuncCachePreferShared);

#define NSTREAMS 8
    cudaStream_t streams[NSTREAMS];

    for (int i = 0; i < NSTREAMS; i++)
        cudaStreamCreate(& (streams[i]));

    allocate((void **) &testim, 10240 * 10240 * sizeof(unsigned char));
    allocate((void **) &filter0, 4*4*2*32 * sizeof(unsigned short));
    allocate((void **) &bias0, 46*46*2*32 * sizeof(unsigned short));
    allocate((void **) &filter1, 4*4*32*32*2 * sizeof(unsigned short));
    allocate((void **) &bias1, 20*20*2*32 * sizeof(unsigned short));
    allocate((void **) &scratch, (OUTPUT_SIZE_0 + OUTPUT_SIZE_1 + OUTPUT_SIZE_2 + OUTPUT_SIZE_3) * sizeof(float));

    load_filters<<<1, 1>>>(filter0, bias0, filter1, bias1);
    if (cudaSuccess != cudaDeviceSynchronize()) {
        printf("failure loading filters\n");
        return -1;
    }
    
    for (int rep = 0; rep < 16; rep++) {
        run_layers<<<40, 256, 0, streams[rep % NSTREAMS]>>>(testim, 10240, 40 * rep, 0, 500, scratch);
    }

    cudaDeviceSynchronize();
    // wait for parent to complete
    return 0;
}
