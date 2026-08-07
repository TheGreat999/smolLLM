#pragma once

#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cfloat>

void callEmbeddingKernel(int* inputTokens, __nv_bfloat16* embeddedInputs, __nv_bfloat16* embedTokensWeight, int input_token_size, int MaxThreadCount);

void callRMSNormKernel(__nv_bfloat16* inputTokens, __nv_bfloat16* normalizedTokens, __nv_bfloat16* normWeights,int input_token_size, int MaxThreadCount);

void callresidualConnectionsKernel(__nv_bfloat16* output, __nv_bfloat16* input, int input_token_size, int MaxThreadCount);

void callmatMul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int m, int k, int n, cublasOperation_t TA, cublasOperation_t TB);

void callRoPE(__nv_bfloat16* Mat);

//WARNING dont let input_token_size be greater than maxThreadCount(1024)
void callSoftMaxKernel(__nv_bfloat16* score, int input_token_size, int maxThreadCount, int sz);

