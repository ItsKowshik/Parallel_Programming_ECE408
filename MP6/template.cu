// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//The global function for the final addition of the scan
__global__ void totalAdd(float *input, float *output, float *sum, int len){
  //Starting ixes
  int t = threadIdx.x;
  int start = 2*blockIdx.x*blockDim.x + t;

  //Shared memory for the final add
  __shared__ float addsum;
  if (t == 0)
  {
    if (blockIdx.x == 0){
      addsum = 0;
    }
    else{
      addsum = sum[blockIdx.x - 1]; 
    }
  }
  //Sync threads befor writing to global memory
  __syncthreads();
  if (start < len){
    output[start] = input[start] + addsum;
  }
  if ((start + blockDim.x) < len){
    output[start + blockDim.x] = input[start + blockDim.x]  + addsum;
  }
}

__global__ void scan(float *input, float *output, int len, int boolScan) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  __shared__ float partialSum[2*BLOCK_SIZE];

  /* Load elements from global */
  int t = threadIdx.x;
  int start;
  int startStride;

  // Determine the starting point and stride based on boolScan
  if (boolScan == 0){
    start = 2*blockIdx.x*blockDim.x + t;
    startStride = blockDim.x;
  }
  else{
    start = 2*blockDim.x*(t+1) - 1;
    startStride = 2*blockDim.x;
  }

  // Load elements into shared memory, handling boundary conditions
  if(start < len){
    partialSum[t] = input[start];
  }
  else{
    partialSum[t] = 0;
  }
  if((start + startStride) < len){
    partialSum[t+blockDim.x] = input[start + startStride];
  }
  else{
    partialSum[t+blockDim.x] = 0;
  }

  /* first scan */
  int stride = 1;
  while (stride < 2*BLOCK_SIZE)
  {
    __syncthreads();
    int ix = (t+1)*stride*2 - 1;
    if ((ix < 2*BLOCK_SIZE) && ((ix-stride) >= 0)){
      partialSum[ix] += partialSum[ix-stride];
    }
    stride *= 2;
  }

  /* second scan */
  stride = BLOCK_SIZE/2;
  while (stride > 0)
  {
    __syncthreads();
    int ix = (t+1)*stride*2 - 1;
    if ((ix + stride) < 2*BLOCK_SIZE){
      partialSum[ix + stride] += partialSum[ix];
    }
    stride /= 2;
  }
  
  /* store back to global */
  __syncthreads();
  int storeStart = 2*blockIdx.x*blockDim.x + t;
  if (storeStart < len){
    output[storeStart] = partialSum[t];
  }
  if ((storeStart + blockDim.x) < len){
    output[storeStart + blockDim.x] = partialSum[t + blockDim.x];
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceScanBlock;
  float *deviceScanSum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanBlock, numElements * sizeof(float)));  // Store the partial sums of each independent block
  wbCheck(cudaMalloc((void **)&deviceScanSum, 2 * BLOCK_SIZE * sizeof(float))); // Store the add-up sums of blocks
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions 
  dim3 dimGrid(ceil(numElements / (2.0*BLOCK_SIZE)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceScanBlock, numElements, 0);
  cudaDeviceSynchronize();

  scan<<<1, dimBlock>>>(deviceScanBlock, deviceScanSum, numElements, 1);
  cudaDeviceSynchronize();

  totalAdd<<<dimGrid, dimBlock>>>(deviceScanBlock, deviceOutput, deviceScanSum, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceScanBlock);
  cudaFree(deviceScanSum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}