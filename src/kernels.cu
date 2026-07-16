#include "kernels.cuh"


/*
__global__ is a cuda syntax its for the function which are host->dev(invoke by host run on dev)
give a matrix embeddedInput of size (inputTokenSize * embeddedDim) 
the blockidx here act as the idx of the token 
and threadidx act as the ith number outtta the 2048 numbs of a embeddedtoken
HARDCODDING 2048 EMBEDDED DIM AND 1024 MAXTHREADSIZE
note here the threadidx goes from [0, 1023]
*/
__global__ void embeddingKernel(int* inputTokens, __nv_bfloat16* embeddedInputs, __nv_bfloat16* embedTokensWeight){
    embeddedInputs[blockIdx.x * 2048 + threadIdx.x] = embedTokensWeight[inputTokens[blockIdx.x] * 2048 + threadIdx.x];
    embeddedInputs[blockIdx.x * 2048 + threadIdx.x + 1024] = embedTokensWeight[inputTokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
}

void callEmbeddingKernel(int* inputTokens, __nv_bfloat16* embeddedInputs, __nv_bfloat16* embedTokensWeight, int input_token_size, int MaxThreadCount){
    embeddingKernel<<<input_token_size, MaxThreadCount>>>(inputTokens, embeddedInputs, embedTokensWeight);

}
    
    