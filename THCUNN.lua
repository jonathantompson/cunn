local ffi = require 'ffi'
local THNN = require 'nn.THNN'

local THCUNN = {}

-- load libTHCUNN
THCUNN.C = ffi.load(package.searchpath('libTHCUNN', package.cpath))

local THCState_ptr = ffi.typeof('THCState*')

function THCUNN.getState()
   return THCState_ptr(cutorch.getState());
end

local THCUNN_h = [[
typedef void THCState;

TH_API void THNN_CudaAbs_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaAbs_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput);

TH_API void THNN_CudaAbsCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage);
TH_API void THNN_CudaAbsCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_CudaClassNLLCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);
TH_API void THNN_CudaClassNLLCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);

TH_API void THNN_CudaDistKLDivCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage);
TH_API void THNN_CudaDistKLDivCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_CudaELU_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          float alpha);
TH_API void THNN_CudaELU_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output,
          float alpha);

TH_API void THNN_CudaHardTanh_updateOutput(
          THCState *state, 
          THCudaTensor *input,
          THCudaTensor *output,
          float min_val,
          float max_val);
TH_API void THNN_CudaHardTanh_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          float min_val,
          float max_val);

TH_API void THNN_CudaL1Cost_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaL1Cost_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput);

TH_API void THNN_CudaLeakyReLU_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          double negval, bool inplace);
TH_API void THNN_CudaLeakyReLU_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          double negval,
          bool inplace);

TH_API void THNN_CudaLogSigmoid_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *buffer);
TH_API void THNN_CudaLogSigmoid_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *buffer);

TH_API void THNN_CudaLogSoftMax_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaLogSoftMax_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output);

TH_API void THNN_CudaLookupTable_accGradParameters(
          THCState *state,
          THIndexTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          float scale,
          bool scaleGradByFreq,
          THIntegerTensor *count,
          THCudaTensor *sorted,
          THCudaTensor *indices);
]]

local preprocessed = string.gsub(THCUNN_h, 'TH_API ', '')

local replacements =
{
   {
      ['THTensor'] = 'THCudaTensor',
      ['THIndexTensor'] = 'THCudaTensor',
      ['THIntegerTensor'] = 'THCudaTensor',
      ['THIndex_t'] = 'float',
      ['THInteger_t'] = 'float'
   }
}

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
   for k,v in pairs(r) do
      s = string.gsub(s, k, v)
   end
   ffi.cdef(s)
end

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_Cuda([%a%d_]+)') do
      t[#t+1] = n
   end
   return t
end

-- build function table
local function_names = extract_function_names(THCUNN_h)

THNN.kernels['torch.CudaTensor'] = THNN.bind(THCUNN.C, function_names, 'Cuda', THCUNN.getState)
torch.getmetatable('torch.CudaTensor').THNN = THNN.kernels['torch.CudaTensor']

return THCUNN
