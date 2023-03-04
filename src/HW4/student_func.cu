//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void getZeroNum(unsigned int* const d_in, unsigned int* d_out, const size_t numElems, int pos){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numElems) return;
    if((d_in[idx]>>pos)&1) return;
    atomicAdd(d_out,1);
}

__global__ void makeIcon(unsigned int* const d_in, unsigned int* d_scan, const size_t numElems, int pos){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numElems) return;
    d_scan[idx] = ((d_in[idx]>>pos)&1)^1;
}

__global__ void getRadixRes(unsigned int* d_inputVals, unsigned int* d_inputPos,
                            unsigned int* d_outputVals, unsigned int* d_outputPos,
                            const size_t numElems, int pos, unsigned int* d_nz, unsigned int* d_scan){
    size_t oid = threadIdx.x + blockDim.x * blockIdx.x;
    if(oid >= numElems) return;
    size_t id;
    if((d_inputVals[oid]>>pos)&1) id = oid + (*d_nz) - d_scan[oid];
    else id = d_scan[oid];
    d_outputPos[id] = d_inputPos[oid];
    d_outputVals[id] = d_inputVals[oid];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  //TODO
  //PUT YOUR SORT HERE

    //0. Definition
    size_t blockSize = 1024;
    size_t gridSize = ceil((float)numElems / blockSize);
    unsigned int* d_nz;
    checkCudaErrors(cudaMalloc(&d_nz, sizeof(unsigned int)));
    thrust::device_vector<unsigned int> d_scan(numElems);

    for(int i = 0; i <= 8 * sizeof(unsigned int); ++i){
        //1. Get Number Of 0
        checkCudaErrors(cudaMemset(d_nz,0, sizeof(unsigned int)));
        getZeroNum<<<gridSize,blockSize>>>(d_inputVals, d_nz, numElems, i);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        //2. Make Icon T/F
        makeIcon<<<gridSize, blockSize>>>(d_inputVals,thrust::raw_pointer_cast(&d_scan[0]),numElems,i);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        //3. Exclusive Scan
        thrust::exclusive_scan(d_scan.begin(),d_scan.end(),d_scan.begin());
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        //4.Radix Result
        getRadixRes<<<gridSize,blockSize>>>(d_inputVals,d_inputPos,d_outputVals,d_outputPos,numElems,i,d_nz,thrust::raw_pointer_cast(&d_scan[0]));
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        //5.Memcpy
        checkCudaErrors(cudaMemcpy(d_inputVals,d_outputVals,numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos,d_outputPos,numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//        break;    //for testing
    }
    checkCudaErrors(cudaFree(d_nz));
}
