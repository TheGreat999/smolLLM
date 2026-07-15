#pragma once

#include <cuda_bf16.h>

void embeddingKernel(int* gpu_inputTokens, __nv_bfloat16* embeddedInputs, __nv_bfloat16* embed_tokens);
