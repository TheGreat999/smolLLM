#pragma once

#include <cuda_bf16.h>

void callEmbeddingKernel(int* inputTokens, __nv_bfloat16* embeddedInputs, __nv_bfloat16* embedTokensWeight, int input_token_size, int MaxThreadCount);
