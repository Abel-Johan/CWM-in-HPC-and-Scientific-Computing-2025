// In this assignment you will write a kernel for vector addition 
// you will also go through generalized processing from of a 
// GPU accelerated application. 
// These are:
//         1) initialize the host and data (allocate memory, load data, ...)
//         2) initialize the device (allocate memory, set its properties, ...)
//         3) transfer data to the device
//         4) run your kernel which will generate some result
//         5) transfer results to the host (eventually)
//         6) clean up (deallocate memory)
//         Run your code
//
// You should follow this assignment in steps mentioned in above list. 
// The TASK 1 correspond to initialization of the host, TASK 2 to 
// initialization of the device and so on.

// NOTE: You should finish your basic "Hello world" assignment first, before 
//       doing this one.

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//----------------------------------------------------------------------
// TASK 4.0: Write your own kernel for vector addition
//
// To calculate the index of the data which given thread should operate
// on use pre-set variables threadIdx, blockIdx, blockDim and gridDim.
//
// Remember that kernel is written from point of view of a single thread,
// i.e. like serial code CPU.


// write your kernel here

//----------------------------------------------------------------------
__global__ void vector_add(int *d_C, int *d_A, int *d_B) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  d_C[index] = d_A[index] + d_B[index];
  // if (index % 524288 == 0) printf("Sanity check %d", d_C[index]);
}

int main(void) {
  //----------------------------------------------------------------------
  // TASK 1: Our overall task is to calculate vector addition. To that end
  //         we have to declare arrays of float which will hold input data,
  //         vectors A and B and also the resulting vector C. All these 
  //         vectors will contain N elements (floats).
  // 
  // First you have to declare variables A, B and C. Remember that dynamically 
  // allocated arrays are expressed with pointers. Allocation of a pointer
  // looks like this: int *pointer_to_int;
  
  // Second step in initialization of the host is allocation of the memory
  // for our data. Allocation on the host could be done by using a 
  // function: void* malloc (size_t size);
  // pointer_to_int = (int*) malloc(size of the array in bytes);
  // The casting of the returned value is necessary because you want both 
  // sides of the expression of the same type. Since malloc returns void*,
  // which you can view as a pointer to a memory without any context, we
  // provide that context by telling the code that what this refers to is
  // actually an int.   
  
  // Last step is to initialize data on the host. We do not load any data
  // because we do not have any, which means you can initialize them to 
  // whatever value you want. However try to initialize them to values 
  // with which you can easily check that your implementation is correct.
  // However try to avoid using values which are same for every element.
  // You can initialize your data for example using a 'for' loop.

  size_t N = 8388608;
  
  // put your code here

  //----------------------------------------------------------------------
  int *h_A, *h_B, *h_C;

  h_A = (int*) malloc(N * sizeof(int));
  h_B = (int*) malloc(N * sizeof(int));
  h_C = (int*) malloc(N * sizeof(int));

  for (size_t f=0; f<N; f++) {
    h_A[f] = f + 1;
    h_B[f] = 2*f + 1;
    h_C[f] = 0;
  }

  //----------------------------------------------------------------------
  // TASK 2: In this task we initialize the GPU, declare variables which 
  //         resided on the GPU and then allocate memory for them.
  //           
  // We must start with device initialization. We do this by using same 
  // process we have used in our "Hello world" code.
  
  // Declaration of variables is no different than what we do for the host
  // it is the location to which the pointer points to which matters.
  
  // Lastly we allocate memory on the device by using cudaMalloc
  // cudaError_t cudaMalloc(void** pointer, size_t size);
  
  // put your code here
  
  //----------------------------------------------------------------------
  int deviceid = 0;
  int devCount;
  cudaGetDeviceCount(&devCount);
  if(deviceid<devCount){
    cudaSetDevice(deviceid);
  }
  else {
    printf("ERROR! Selected device is not available\n");
    return(1);
  }

  int *d_A, *d_B, *d_C;

  cudaMalloc(&d_A, N * sizeof(int));
  cudaMalloc(&d_B, N * sizeof(int));
  cudaMalloc(&d_C, N * sizeof(int));
  //----------------------------------------------------------------------
  // TASK 3: Here we would like to copy the data from the host to the device
  
  // To do that we will use function 'cudaMemcpy'
  // cudaError_t cudaMemcpy(destination, source, size, direction);
  // where direction is either from the host to the device
  // 'cudaMemcpyHostToDevice' or from the device to the host 
  // 'cudaMemcpyDeviceToHost'.

  // put your code here

  //----------------------------------------------------------------------
  cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

  //----------------------------------------------------------------------
  // TASK 4.0: To write your vector addition kernel. Full task is above.
  //----------------------------------------------------------------------

  //----------------------------------------------------------------------
  // TASK 4.1: Now having data on the device and having a kernel for vector
  //           addition we would like to execute that kernel. 
  //
  // You can choose what ever grid configuration you desire, but take into 
  // account that, unless you have written the kernel otherwise, it cannot
  // handle data sizes which are not equal to 
  // (number of threads per block)*(number of blocks) == N !
  // In other words if N=200 and you are using 25 threads per block
  // you must launch your kernel with 8 blocks.
  
  // put your code here
  
  //----------------------------------------------------------------------
  dim3 Gd(256, 8, 8);
  dim3 Bd(8, 8, 8);
  vector_add<<<Gd, Bd>>>(d_C, d_A, d_B);
  //----------------------------------------------------------------------
  // TASK 5: Transfer data to the host.
  
  // put your code here

  //----------------------------------------------------------------------
  cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
  
  if(N>1000){
	  printf("Check:\n");
	  for(int f=0; f<1000; f++){
		  printf("Is %d + %d = %d?\n", h_A[f], h_B[f], h_C[f]);
	  }
  }
  
  
  //----------------------------------------------------------------------
  // TASK 6: Free allocated resources.
  //
  // To do this on the device use cudaFree();
  
  // put your code here

  //----------------------------------------------------------------------
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  // TASK 7: Run your code
  return(0);
}

