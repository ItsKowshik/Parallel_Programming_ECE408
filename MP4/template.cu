#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 4
#define MASK_WIDTH 3

//@@ Define constant memory for device kernel here
__constant__ float M[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float input_ds[TILE_WIDTH+2][TILE_WIDTH+2][TILE_WIDTH+2];

 int tx = threadIdx.x;
 int ty = threadIdx.y;
 int tz = threadIdx.z;
 int bx = blockIdx.y * TILE_WIDTH + ty;
 int by = blockIdx.x * TILE_WIDTH + tx;
 int bz = blockIdx.z * TILE_WIDTH + tz;

 int bx_i = bx - 1;       
 int by_i = by - 1;
 int bz_i = bz - 1;

 float sum = 0.0f;

 // Set the halo elements to 0.
 if((bx_i >= 0) && (bx_i < y_size) && (by_i >= 0) && (by_i < x_size) && (bz_i >= 0) && (bz_i < z_size)){
     input_ds[tz][ty][tx] = input[bz_i * x_size * y_size + bx_i * x_size + by_i];
   }
 else{
     input_ds[tz][ty][tx] = 0.0f;
   }
  __syncthreads();

  if((tx < TILE_WIDTH) && (ty < TILE_WIDTH) && (tz < TILE_WIDTH)){
    for(int i = 0; i < MASK_WIDTH; i++){
      for(int j = 0; j < MASK_WIDTH; j++){
        for(int k = 0; k < MASK_WIDTH; k++){
          sum += M[i][j][k] * input_ds[i+tz][j+ty][k+tx];
        }
      }
    }
    if((bx < y_size) && (by < x_size) && (bz < z_size)){
    output[bz * x_size * y_size + bx * x_size + by] = sum;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =(float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int input_size = (inputLength - 3)*sizeof(float);
  cudaMalloc((void **)&deviceInput, input_size);
  cudaMalloc((void **)&deviceOutput, input_size);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], input_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(M, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);
  dim3 dimGrid(ceil(x_size / (1.0 * TILE_WIDTH)),ceil(y_size / (1.0 * TILE_WIDTH)), ceil(z_size / (1.0 * TILE_WIDTH)));
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, input_size, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
