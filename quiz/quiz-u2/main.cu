#include "timer.h"
#include <cstdio>

#define ARRAY_SIZE 100
#define NUM_THREADS 10000000
#define BLOCK_WIDTH 1000

int h_array[ARRAY_SIZE];

__global__ void increament_native(int *g){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = i % ARRAY_SIZE;
    g[i] = g[i] + 1;
}

__global__ void increament_atomic(int *g){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = i % ARRAY_SIZE;
    atomicAdd(&g[i],1);
}

inline void print_array(int* array,int size){
    printf("(");
    for(int i=0;i<size;++i){
        if(i) printf(",");
        printf("%d",array[i]);
    }
    printf(")\n");
}

int main() {
    GpuTimer timer;
    printf("%d total threads in %d blocks writing into %d array elements\n",
           NUM_THREADS,NUM_THREADS/BLOCK_WIDTH,ARRAY_SIZE);

    //host
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    //device
    int* d_array;
    cudaMalloc((void **)&d_array,ARRAY_BYTES);
    cudaMemset((void *)d_array, 0, ARRAY_BYTES);

    //lauch kernal
    timer.Start();
    increament_atomic<<<NUM_THREADS/BLOCK_WIDTH,BLOCK_WIDTH>>>(d_array);
    timer.Stop();

    //copy back
    cudaMemcpy(h_array,d_array,ARRAY_BYTES,cudaMemcpyDeviceToHost);
    print_array(h_array,ARRAY_SIZE);
    printf("Time elapsed = %g ms\n",timer.Elapsed());

    cudaFree(d_array);
    return 0;
}

/***  GPU: 3070Ti   MUCH FASTER THAN CPU!!!
*  threadNum   ArraySize   increase   Result   TimeElapsed(ms)
 * 1e6         1e6         Native     ok       0.016384
 * 1e6         1e6         Atomic     ok       0.012288
 * 1e6         100         Native     wrong    0.013312
 * 1e6         100         Atomic     ok       0.095232
 * 1e7         100         Atomic     ok       0.908288
*/
