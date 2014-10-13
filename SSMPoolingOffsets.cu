
//for each input plane there will be an X-offset and a Y-offset plane (i.e. output_n = 2*input_n)

//no-overlap 
__global__ void output_kernel(float *input, float *output, float *softmax, 
                        float *gridX, float *gridY,
                        int input_n, int input_h, int input_w, int output_h, int output_w,
                        int kH, int kW)
{
    float* ptr_input_plane = input + blockIdx.x * input_w * input_h;
    float* ptr_softmax_plane = softmax + blockIdx.x * input_w * input_h;
    float* ptr_output_plane = output + 2*blockIdx.x * output_w * output_h;
    
    float* ptr_output_Xplane = ptr_output_plane; 
    float* ptr_output_Yplane = ptr_output_plane + output_w * output_h;

    int yout = threadIdx.y; 
    int xout_step = blockDim.x;  
    int yout_step = blockDim.y;  
    
    int xin_start = threadIdx.x * kW;
    int yin_start = threadIdx.y * kH; 
    int xin_step = blockDim.x * kW;
    int yin_step = blockDim.y * kH;
    int xin_end = (input_w/kW) * kW; 
    int yin_end = (input_h/kH) * kH; 
    
    for (int yin = yin_start; yin < yin_end; yin += yin_step) {
        int xout = threadIdx.x;
        for (int xin = xin_start; xin < xin_end; xin += xin_step) {
        float* ptr_input = ptr_input_plane + xin + yin * input_w;
        float* ptr_softmax = ptr_softmax_plane + xin + yin * input_w;
        float* ptr_output_X = ptr_output_Xplane + xout + yout*output_w;  
        float* ptr_output_Y = ptr_output_Yplane + xout + yout*output_w; 
        float pool_sum = 0;
        float offsetX = 0; 
        float offsetY = 0; 

          if (xout < output_w && yout < output_h) {

          for (int ky = 0; ky < kH && yin + ky < input_h; ky++) { 
            for (int kx = 0; kx < kW && xin + kx < input_w; kx++) {
              float* ptr_input_pool = ptr_input + kx + ky * input_w;
              pool_sum += exp(*ptr_input_pool);  
            }
          }
          
          for (int ky = 0; ky < kH && yin + ky < input_h; ky++) { 
            for (int kx = 0; kx < kW && xin + kx < input_w; kx++) {
              float* ptr_input_pool = ptr_input + kx + ky * input_w;
              float* ptr_softmax_pool = ptr_softmax + kx + ky * input_w;
              float sm_val = exp(*ptr_input_pool)/pool_sum; 
              *ptr_softmax_pool = sm_val; 
              offsetX += gridX[kx] * sm_val;  
              offsetY += gridY[ky] * sm_val;
            }
          }
       
            *ptr_output_X = offsetX;
            *ptr_output_Y = offsetY;
          }

        xout += xout_step;  
        }
    yout += yout_step; 
    }

}

__global__ void gradInput_kernel(float *gradInput, float *gradOutput, float *softmax, 
                             float *gridX, float *gridY,
                             int input_n, int input_h, int input_w, int output_h, int output_w,
                             int kH, int kW)
{

    //select the block
    float* ptr_gradInput_plane = gradInput + blockIdx.x * input_w * input_h;
    float* ptr_softmax_plane = softmax + blockIdx.x * input_w * input_h;
    float* ptr_gradOutput_plane = gradOutput + 2*blockIdx.x * output_w * output_h;
    float* ptr_gradOutput_Xplane = ptr_gradOutput_plane; 
    float* ptr_gradOutput_Yplane = ptr_gradOutput_plane + output_w * output_h;
    
    int xin_start = threadIdx.x * kW;
    int yin_start = threadIdx.y * kH; 
    int xin_step = blockDim.x * kW;
    int yin_step = blockDim.y * kH;
    int xin_end = (input_w/kW)*kW; 
    int yin_end = (input_h/kH)*kH; 
   
    int yout = threadIdx.y; 
    int xout_step = blockDim.x;  
    int yout_step = blockDim.y;  

    for (int yin = yin_start; yin < yin_end; yin += yin_step) {
        int xout = threadIdx.x;
        for (int xin = xin_start; xin < xin_end; xin += xin_step) {
        float* ptr_gradInput = ptr_gradInput_plane + xin + yin * input_w;
        float* ptr_gradOutput_X = ptr_gradOutput_Xplane + xout + yout*output_w;  
        float* ptr_gradOutput_Y = ptr_gradOutput_Yplane + xout + yout*output_w; 
        float* ptr_softmax = ptr_softmax_plane + xin + yin * input_w;
        float pool_sum_X = 0;
        float pool_sum_Y = 0; 

          for (int ky = 0; ky < kH && yin + ky < input_h; ky++) { 
            for (int kx = 0; kx < kW && xin + kx < input_w; kx++) {
              float* ptr_softmax_pool = ptr_softmax + kx + ky * input_w;
              pool_sum_X += (*ptr_softmax_pool) * gridX[kx];  
              pool_sum_Y += (*ptr_softmax_pool) * gridY[ky];  
            }
          }
          
         for (int ky = 0; ky < kH && yin + ky < input_h; ky++) { 
            for (int kx = 0; kx < kW && xin + kx < input_w; kx++) {
              float* ptr_gradInput_pool = ptr_gradInput + kx + ky * input_w;
              float* ptr_softmax_pool = ptr_softmax + kx + ky * input_w;
              *ptr_gradInput_pool = (*ptr_softmax_pool) * ((*ptr_gradOutput_X * (gridX[kx] - pool_sum_X)) 
                                                          +(*ptr_gradOutput_Y * (gridY[ky] - pool_sum_Y)));
            }
          }
        xout += xout_step; 
        }
    yout += yout_step; 
    }
}

static int cunn_SSMPoolingOffsets_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *softmax = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "softmax", "torch.CudaTensor");
  THCudaTensor *gridX = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gridX", "torch.CudaTensor");
  THCudaTensor *gridY = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gridY", "torch.CudaTensor");
  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");

  float *output_data;
  float *input_data;
  float *softmax_data; 
  float *gridX_data;
  float *gridY_data;

  long nInputCols = input->size[3];
  long nInputRows = input->size[2];
  long nInputPlane = input->size[1];
  long nbatch = input->size[0];
  long nOutputCols = nInputCols/kW;
  long nOutputRows = nInputRows/kH;

  luaL_argcheck(L, input->size[1] == nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

  input = THCudaTensor_newContiguous(input);
  input_data = THCudaTensor_data(input);

  THCudaTensor_resize4d(output, nbatch, 2*nInputPlane, nOutputRows, nOutputCols);
  THCudaTensor_resize4d(softmax, nbatch, nInputPlane, nInputRows, nInputCols);

  output_data = THCudaTensor_data(output);
  softmax_data = THCudaTensor_data(softmax);
  gridX_data = THCudaTensor_data(gridX);
  gridY_data = THCudaTensor_data(gridY);

  // cuda blocks & threads:
  dim3 blocks(nInputPlane*nbatch,1);
  dim3 threads(32,8);

  // run maxpool kernel
  output_kernel <<<blocks, threads>>> (input_data, output_data, softmax_data, gridX_data, gridY_data,
                                   nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols, kH, kW);
  // clean
  THCudaTensor_free(input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SSMPoolingOffsets.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int cunn_SSMPoolingOffsets_updateGradInput(lua_State *L)
{
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *softmax = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "softmax", "torch.CudaTensor");
  THCudaTensor *gridX = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gridX", "torch.CudaTensor");
  THCudaTensor *gridY = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gridY", "torch.CudaTensor");
  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");

  float *gradInput_data;
  float *gradOutput_data;
  float *softmax_data;
  float *gridX_data;
  float *gridY_data;

  long nInputCols = input->size[3];
  long nInputRows = input->size[2];
  long nInputPlane = input->size[1];
  long nbatch = input->size[0];
  long nOutputCols = gradOutput->size[3];
  long nOutputRows = gradOutput->size[2];

  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_zero(gradInput);  

  gradOutput_data = THCudaTensor_data(gradOutput);
  gradInput_data = THCudaTensor_data(gradInput);
  softmax_data = THCudaTensor_data(softmax);
  gridX_data = THCudaTensor_data(gridX);
  gridY_data = THCudaTensor_data(gridY);

  dim3 blocks(nInputPlane*nbatch,1);
  dim3 threads(32,8);

  // run updateGradInput kernel
  gradInput_kernel<<<blocks, threads>>>
	(gradInput_data, gradOutput_data, softmax_data,
	 gridX_data, gridY_data, 
     nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols, kH, kW);
  

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SSMPoolingOffsetssampling.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static const struct luaL_Reg cunn_SSMPoolingOffsets__ [] = {
  {"SSMPoolingOffsets_updateOutput", cunn_SSMPoolingOffsets_updateOutput},
  {"SSMPoolingOffsets_updateGradInput", cunn_SSMPoolingOffsets_updateGradInput},
  {NULL, NULL}
};

static void cunn_SSMPoolingOffsets_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SSMPoolingOffsets__, "nn");
  lua_pop(L,1);
}

