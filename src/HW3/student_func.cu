/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__ void getMaxOrMin(const float * d_in, float * d_out, bool opMax, const size_t tot){
    extern __shared__ float sh_mem[];
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t tid = threadIdx.x;
    if(idx >= tot) sh_mem[tid] = d_in[0];
    else sh_mem[tid] = d_in[idx];
    __syncthreads();
    for(int s = blockDim.x/2;s>0;s>>=1){
        if(tid < s){
            if(opMax) sh_mem[tid] = max(sh_mem[tid],sh_mem[tid+s]);
            else sh_mem[tid] = min(sh_mem[tid],sh_mem[tid+s]);
        }
        __syncthreads();
    }
    if(!tid) d_out[blockIdx.x] = sh_mem[0];
}

__global__ void makeHistogram(const float * d_in, unsigned unsigned int * d_out, float lumMin, float lumRange, const size_t numBins,const size_t tot){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= tot) return;
    int bel = (d_in[idx] - lumMin) / lumRange * numBins;
    if(bel == numBins) --bel;
    atomicAdd(&d_out[bel],1);
}

__global__ void blellochScan(unsigned int * d_in, const size_t length){
    extern __shared__ unsigned unsigned int sh_res[];
    size_t idx = threadIdx.x;
    sh_res[idx] = d_in[idx];
    __syncthreads();
    for(int s = 1;s<length;s<<=1){
        if(!((idx+1)%(s<<1))){
            sh_res[idx] += sh_res[idx - s];
        }
        __syncthreads();
    }
    if(idx == length - 1) sh_res[idx] = 0;
    __syncthreads();
    for(int s = (length>>1);s > 0;s>>=1){
        if(idx&&!((idx+1)%(s<<1))){
            sh_res[idx] += sh_res[idx - s];
            sh_res[idx - s] = sh_res[idx] - sh_res [idx - s];
        }
        __syncthreads();
    }
    d_in[idx] = sh_res[idx];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    //Step 1: get Max&Min
    size_t blockSize = 1024;
    size_t gridSize = exp2(ceil(log2(numRows * numCols / blockSize)));
    size_t tot = numRows * numCols;

    //get max
    float * d_im;
    checkCudaErrors(cudaMalloc(&d_im,sizeof(float) * gridSize));
    getMaxOrMin<<<gridSize,blockSize,sizeof(float) * blockSize>>>(d_logLuminance,d_im, true,tot);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    float * d_max;
    checkCudaErrors(cudaMalloc(&d_max,sizeof(float)));
    getMaxOrMin<<<1,gridSize, sizeof(float) * gridSize>>>(d_im,d_max, true,tot);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //get min
    getMaxOrMin<<<gridSize,blockSize,sizeof(float) * blockSize>>>(d_logLuminance,d_im, false,tot);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    float * d_min;
    checkCudaErrors(cudaMalloc(&d_min,sizeof(float)));
    getMaxOrMin<<<1,gridSize, sizeof(float) * gridSize>>>(d_im,d_min, false,tot);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //memcpy
    checkCudaErrors(cudaMemcpy(&min_logLum,d_min,sizeof(float),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&max_logLum,d_max,sizeof(float),cudaMemcpyDeviceToHost));

    printf("max=%.2f  min=%.2f\n",max_logLum,min_logLum);
    //Step 2: getRange
    float lumRange = max_logLum - min_logLum;

    //Step 3: getHistogram
    makeHistogram<<<gridSize,blockSize>>>(d_logLuminance,d_cdf,min_logLum,lumRange,numBins,tot);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //Step 4: Scan
    blellochScan<<<1,numBins, sizeof(unsigned int) * numBins>>>(d_cdf,numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_max));
    checkCudaErrors(cudaFree(d_min));
    checkCudaErrors(cudaFree(d_im));
}
