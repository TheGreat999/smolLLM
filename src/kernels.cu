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
    
    
/*
__shared__ shares the variable with all the threads within a block
we areusing float here so we dont loose precision
here we are initially storing square of i and i + 1024 in rms_vals this saves space and time of one iteration
__syncthreads() block the execution till all threads reached this point within a block
*/
__global__ void RMSNormKernel(__nv_bfloat16* inputTokens, __nv_bfloat16* normalizedTokens, __nv_bfloat16* normWeights){
    __shared__ float rms_vals[1024];
    int idx = threadIdx.x + blockIdx.x * 2048;
    rms_vals[threadIdx.x] = ((float)inputTokens[idx] * (float)inputTokens[idx]) + ((float)inputTokens[idx + 1024] * (float)inputTokens[idx + 1024]);
    __syncthreads();
    #pragma unroll
    for(int stride = 512; stride >= 64; stride >>= 1){
        if(threadIdx.x < stride){
            rms_vals[threadIdx.x] += rms_vals[threadIdx.x + stride];
        }
        __syncthreads();
    }

    /*IMPORTANT SHIT HERE
        here now the number of working threads is 32 they all are in the same warp 
        and threads in the same warp are executed synchronously so we dont need to add __syncthreads();
        __shfl_down_sync this essensially let the thread copy its neighboring threads register value in its own register
        __shfl_down_sync(mask, variable, offset)
        mask define which register will execute the shfl function
        variable is the name of the var which will be copied
        offset is the idx of thread relative to current ie idx = thread.x + offset
    */
    if(threadIdx.x < 32){
        float sum = rms_vals[threadIdx.x] + rms_vals[threadIdx.x + 32];

        #pragma unroll
        for(int offset = 16; offset > 0; offset >>= 1){
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if(threadIdx.x == 0){
            rms_vals[0] = sqrt(sum/2048.0 + 1.0e-5);
        }
    }

    __syncthreads();

    normalizedTokens[idx] = (__nv_bfloat16)((float)(inputTokens[idx]) / rms_vals[0] * (float)normWeights[threadIdx.x]);
    normalizedTokens[idx + 1024] = (__nv_bfloat16)((float)(inputTokens[idx + 1024]) / rms_vals[0] * (float)normWeights[threadIdx.x + 1024]);

}

void callRMSNormKernel(__nv_bfloat16* inputTokens, __nv_bfloat16* normalizedTokens, __nv_bfloat16* normWeights,int input_token_size, int MaxThreadCount){
    RMSNormKernel<<<input_token_size, MaxThreadCount>>>(inputTokens, normalizedTokens, normWeights);
}