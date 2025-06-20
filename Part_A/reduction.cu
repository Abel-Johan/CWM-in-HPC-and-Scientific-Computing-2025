#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <curand.h>

#define NUM_THREADS 1024 // max of 1024 threads per block on the A100

__global__ void reduction(float *d_input, float *d_output)
{
    // Allocate shared memory

    __shared__  float smem_array[NUM_THREADS];
    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    smem_array[tid] = d_input[tid];
    __syncthreads();

    // next, we perform binary tree reduction

    for (int d = blockDim.x/2; d > 0; d /= 2) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d)  smem_array[tid] += smem_array[tid+d];
    }

    // finally, first thread puts result into global memory
    // this is because the final sum will be in element zero!
    if (tid==0) {
      d_output[blockIdx.x] = smem_array[0];
    }

}

__global__ void reduction_final(float *d_input, float *d_output)
{
    // Allocate shared memory

    __shared__  float smem_array[NUM_THREADS];
    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    smem_array[tid] = d_input[tid];
    __syncthreads();

    // next, we perform binary tree reduction

    for (int d = blockDim.x/2; d > 0; d /= 2) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d)  smem_array[tid] += smem_array[tid+d];
    }

    // finally, first thread puts result into global memory
    // this is because the final sum will be in element zero!
    if (tid==0) {
      d_output[0] = smem_array[0];
    }

}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
    // Declare variables and arrays to store data
    int num_els, num_blocks, mem_size, num_threads;

    float *h_data;
    float *d_input, *d_output;

    // Ask user for how many numbers to implement reduction on
    printf("Enter number of elements: ");
    scanf("%d", &num_els);
    printf("\n");

    // initialise card
    int deviceid = 0;
    int devCount;
    cudaGetDeviceCount(&devCount);
    if(deviceid<devCount){
      cudaSetDevice(deviceid);
    }
    else {
      printf("ERROR! Selected device is not available\n");
      return(2);
    }

    // Detect whether number of elements is a power of two or not
    int checker = num_els; // used to iterate and check whether num_els is a power of two or not
    int exponent_of_two = 0; // how many powers of two constitute num_els
    // e.g. if exponent_of_two = 4, then num_els has a factor of 2^4 = 16.
    while (1) {
      // if it indeed is a power of two...
      if (checker / 2 == 1) {
        // of 1024 or more, then let num_threads be 1024, i.e. one or more blocks used depending on num_els
        if (num_els >= 1024) {
	  num_threads = NUM_THREADS;
        // of less than 1024, then let num_threads = num_els, i.e. one block only
        } else {
	  num_threads = num_els;
	}
	break;
      // if it is a generic even number, then divide by two and record exponent_of_two
      } else if (checker % 2 == 0) {
        checker /= 2;
        exponent_of_two++;
        continue;
      // if it is not a power of two...
      } else {
        // let num_threads = highest power-of-two factor
        // this will mean num_blocks will be an odd number
        num_threads = 1;
	for (int i = 0; i < exponent_of_two; i++) {
	  num_threads *= 2;
	}
        break;
      }
    }
    
    num_blocks = (int) num_els/num_threads;
    mem_size    = sizeof(float) * num_els;

    // allocate host memory to store the input data
    h_data = (float*) malloc(mem_size);

   /* RELIC FROM GIVEN CODE
    for(int i = 0; i < num_els; i++) {
        h_data[i] = ((float)rand()/(float)RAND_MAX);
    }
    RELIC FROM GIVEN CODE */

    // allocate device memory input and output arrays
    cudaMalloc((void**)&d_input, mem_size);
    cudaMalloc((void**)&d_output, sizeof(float)); // because the output is only one float at element zero!


    // Generate randoms using CURAND
    // Declare variable
    curandGenerator_t gen;
    // Create RNG
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the generator options
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    // Generate the randoms
    // This will create <num_els> amount of random numbers and put it into d_input
    // Draw from a normal distribution with mean 5 and std 3
    if (curandGenerateNormal(gen, d_input, num_els, 5.0f, 3.0f) != CURAND_STATUS_SUCCESS) {
        printf("Something wrong :(");
        return(1);
    }


    /* RELIC FROM GIVEN CODE
    // copy host memory to device input array

    cudaMemcpy(d_input, h_data, mem_size, cudaMemcpyHostToDevice);
    RELIC FROM GIVEN CODE */


    // execute the kernel
    // reduce across threads in a block
    reduction<<<num_blocks,NUM_THREADS>>>(d_input,d_output);
    // reduce across the results from each block
    // store the result back into d_input
    reduction_final<<<1,num_blocks>>>(d_output, d_input);

    // copy result from device to host

    cudaMemcpy(h_data, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    // check results

    printf("reduction error = %f\n",h_data[0]/num_els);

    // cleanup memory

    free(h_data);
    cudaFree(d_input);
    cudaFree(d_output);

    // CUDA exit -- needed to flush printf write buffer

    cudaDeviceReset();
}
