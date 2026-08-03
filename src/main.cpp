#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map> 
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "json.hpp"
#include "kernels.cuh"

/*
    This prevent implicit convertion example:
        json j = {{"age", 20}}; 
        int age = j["age"] ❌
        int age = j["Age"].get<int>() ✔️
*/
#define JSON_USE_IMPLICIT_CONVERSIONS 0
using json = nlohmann::json;

using std::cout, std::cerr, std::cin, std::to_string;

#define TIMER_START(name) \
    auto name##_start = std::chrono::steady_clock::now()

#define TIMER_END(name) \
    std::cout << #name << ": " \
              << std::chrono::duration_cast<std::chrono::milliseconds>( \
                     std::chrono::steady_clock::now() - name##_start) \
                     .count() \
              << " ms\n"

              
constexpr int B2MB = 1024 * 1024;

// These are need to configure for each model
//TODO: add coommments for each explaining their meaning
constexpr int N_LAYERS = 16;
constexpr int EMBEDDING_DIMENSIONS = 2048;
constexpr int VOCABULURY = 128256;
constexpr int MAX_TOKENS = 2048;
constexpr int MAX_INPUT_TOKENS = 512;
constexpr int Q_HEADS = 32;
constexpr int KV_DIM = 512;
constexpr int KV_HEADS = 8;
constexpr int GQA_Q_TO_K_RATIO = Q_HEADS / KV_HEADS;
constexpr int HEAD_DIM = EMBEDDING_DIMENSIONS / Q_HEADS;



struct Weights{
    //Its only store the location of the weights on the GPU
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layer_norm[N_LAYERS]; // RMS Norm 1
    __nv_bfloat16 *k_proj[N_LAYERS]; // Key
    __nv_bfloat16 *q_proj[N_LAYERS]; // Query
    __nv_bfloat16 *v_proj[N_LAYERS]; // Value
    __nv_bfloat16 *o_proj[N_LAYERS]; // Output
    __nv_bfloat16 *post_attn_layer_norms[N_LAYERS]; // RMS Norm 2
    __nv_bfloat16 *mlp_up_proj[N_LAYERS]; // 2048 -> 8192 thingy
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS]; // the middle magic thingy
    __nv_bfloat16 *mlp_down_proj[N_LAYERS]; // 8192 -> 2048 thingy
    __nv_bfloat16 *final_norm; // RMS Norm 3
};

static Weights weights;

int smCount, MaxThreadCount;
int checkGPUStatus(){
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }
    cout << "==========================\n";
    cout << "Device Info\n";
    cout << "==========================\n";
    cudaDeviceSynchronize();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    // This is the version of the supported features not some quantity
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B2MB << " MB\n";
    smCount = prop.multiProcessorCount;
    std::cout << "SM count: " << smCount << "\n";
    MaxThreadCount = prop.maxThreadsPerBlock;
    std::cout << "Max threads per block: " << MaxThreadCount << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B2MB << "MB, total memory: " << total_mem / B2MB << "MB\n\n";
    return 0;
}

int RMS_layer(__nv_bfloat16* inputTokens, __nv_bfloat16** normalizedTokens, __nv_bfloat16* normWeights, int input_token_size){


    // cout << "==========================\n";
    // cout << "Input layer Normalization \n";
    // cout << "==========================\n";
    // TIMER_START(RMSNorm1Time);
    cudaMalloc(normalizedTokens, sizeof(__nv_bfloat16) * input_token_size * 2048);

    callRMSNormKernel(inputTokens, *normalizedTokens, normWeights, input_token_size, MaxThreadCount);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        cerr<<cudaGetErrorName(err)<<'\n'<<cudaGetErrorString(err) << '\n';
        return 1;
    }

    // TIMER_END(RMSNorm1Time);
    // cout << "Normalized Input layer successfully\n\n";

    return 0;
    
}

int cpyToGPU(void** modelWeights, uint64_t maxOffset, std::ifstream& model){
    TIMER_START(Devalloc);
    cudaMalloc(modelWeights, maxOffset);
    TIMER_END(Devalloc);

    TIMER_START(Hostalloc);
    char* temp_cpu_buffer;
    cudaHostAlloc(&temp_cpu_buffer, maxOffset, cudaHostAllocDefault);
    TIMER_END(Hostalloc);

    TIMER_START(fileRead);
    model.read(temp_cpu_buffer, maxOffset);
    TIMER_END(fileRead);

    TIMER_START(Host2Devcpy);
    cudaError_t err = cudaMemcpy(*modelWeights, temp_cpu_buffer, maxOffset, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        cout<<"There was a error when copying weights to the GPU\n";
        cerr<<cudaGetErrorName(err)<<'\n'<<cudaGetErrorString(err) << '\n';
        model.close();
        return 1;
    }
    cudaDeviceSynchronize();
    TIMER_END(Host2Devcpy);
    cudaFreeHost(temp_cpu_buffer);
    cout<<"Copied "<<maxOffset / B2MB <<" MB of weights to GPU successfully\n";
    model.close();
    return 0;
}

int loadWeights(){

    if(checkGPUStatus() != 0)return 1;
    std::ifstream model("./models/Llama-3.2-1B-Instruct/model.safetensors", std::ios_base::binary);
    if(!model.is_open()){
        cout<<"Could not open model file\n";
        model.close();
        return 1;
    }
    // cout<<"Model file opened successfully\n";


    uint64_t header_size;
    model.read(reinterpret_cast<char *> (&header_size), 8);
    std::string header(header_size, '\0');
    model.read(header.data(), header_size);
    json header_json = json::parse(header);
    std::unordered_map<std::string, uint64_t> offsets;
    uint64_t maxOffset = 0;
    for(auto& [key, val] : header_json.items()){
        if(key == "__metadata__")continue;
        uint64_t offsetEnd = val["data_offsets"].at(1).get<uint64_t>();
        if(offsetEnd > maxOffset){
            maxOffset = offsetEnd;
        }
        offsets[key] = val["data_offsets"].at(0).get<uint64_t>();
    }

    cout << "==========================\n";
    cout << "Loading Weights\n";
    cout << "==========================\n";
    void* modelWeights; // This pointer will track the location of the weights;
    // REMEBER TO PASS THE POINTER AS A REFF
    if(cpyToGPU(&modelWeights, maxOffset, model) != 0){
        return 1;
    }

    weights.embed_tokens = (__nv_bfloat16 *)((char *) modelWeights + offsets["model.embed_tokens.weight"]);
    for(int layer = 0; layer < N_LAYERS; layer++){
        weights.input_layer_norm[layer] = (__nv_bfloat16 *) ((char *)modelWeights + offsets["model.layers." +to_string(layer) + ".input_layernorm.weight"]);
        weights.k_proj[layer] = (__nv_bfloat16 *) ((char *)modelWeights + offsets["model.layers." +to_string(layer) + ".self_attn.k_proj.weight"]);
        weights.q_proj[layer] = (__nv_bfloat16 *) ((char *)modelWeights + offsets["model.layers." +to_string(layer) + ".self_attn.q_proj.weight"]);
        weights.v_proj[layer] = (__nv_bfloat16 *) ((char *)modelWeights + offsets["model.layers." +to_string(layer) + ".self_attn.v_proj.weight"]);
        weights.o_proj[layer] = (__nv_bfloat16 *) ((char *)modelWeights + offsets["model.layers." +to_string(layer) + ".self_attn.o_proj.weight"]);
        weights.post_attn_layer_norms[layer] = (__nv_bfloat16 *) ((char *)modelWeights + offsets["model.layers." +to_string(layer) + ".post_attention_layernorm.weight"]);
        weights.mlp_down_proj[layer] = (__nv_bfloat16 *) ((char *)modelWeights + offsets["model.layers." +to_string(layer) + ".mlp.down_proj.weight"]);
        weights.mlp_gate_proj[layer] = (__nv_bfloat16 *) ((char *)modelWeights + offsets["model.layers." +to_string(layer) + ".mlp.gate_proj.weight"]);
        weights.mlp_up_proj[layer] = (__nv_bfloat16 *) ((char *)modelWeights + offsets["model.layers." +to_string(layer) + ".mlp.up_proj.weight"]);
    }
    weights.final_norm = (__nv_bfloat16 *)((char *) modelWeights + offsets["model.norm.weight"]);
    cout<<"Weights pointers marked successfully\n\n";
    return 0;

}

int loadTokensandEmbedding(std::vector<int> & input_tokens, __nv_bfloat16** embeddedInputs, int input_token_size, int embedding_size){
    if(input_token_size == 0){
        cerr << "Input tokens are empty\n";
        return 1;
    }
    cout << "==========================\n";
    cout << "Embedding Tokens\n";
    cout << "==========================\n";
    TIMER_START(EmbeddingTime);
    int* gpu_inputTokens;
    cudaMalloc(&gpu_inputTokens, sizeof(int) * input_token_size);
    cudaMemcpy(gpu_inputTokens, input_tokens.data(), sizeof(int) * input_token_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // __nv_bfloat16* embeddedInputs;
    cudaMalloc(embeddedInputs, embedding_size);

    /*This is how you envoke a kernel(function) on a gpu
        kernel_name<<<number of blocks, thread per block>>>(attribs);
        thread per block should not exccedd MaxThreadCount
    */
    callEmbeddingKernel(gpu_inputTokens, *embeddedInputs, weights.embed_tokens, input_token_size, MaxThreadCount);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        cerr<<cudaGetErrorName(err)<<'\n'<<cudaGetErrorString(err) << '\n';
        return 1;
    }

    cudaFree(gpu_inputTokens);
    TIMER_END(EmbeddingTime);
    cout << "Embedding successfull\n\n";

    return 0;
}

int attentionPass(int layer, nv_bfloat16* normalizedTokens, int input_token_size, int embedding_size, nv_bfloat16* outputScore){

    nv_bfloat16* k_proj;
    cudaMalloc(&k_proj, input_token_size * KV_DIM * sizeof(__nv_bfloat16));
    callmatMul(normalizedTokens, weights.k_proj[layer], k_proj, input_token_size, EMBEDDING_DIMENSIONS, KV_DIM, CUBLAS_OP_N, CUBLAS_OP_N);

    nv_bfloat16* q_proj;
    cudaMalloc(&q_proj, embedding_size);
    callmatMul(normalizedTokens, weights.q_proj[layer], q_proj, input_token_size, EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS, CUBLAS_OP_N, CUBLAS_OP_N);

    nv_bfloat16* v_proj;
    cudaMalloc(&v_proj, input_token_size * KV_DIM * sizeof(__nv_bfloat16));
    callmatMul(normalizedTokens, weights.v_proj[layer], v_proj, input_token_size, EMBEDDING_DIMENSIONS, KV_DIM, CUBLAS_OP_N, CUBLAS_OP_N);

    nv_bfloat16* attentionScore;
    cudaMalloc(&attentionScore, input_token_size * input_token_size * sizeof(__nv_bfloat16));
    
    // nv_bfloat16* outputScore;
    cudaMalloc(&outputScore, input_token_size * input_token_size * sizeof(__nv_bfloat16));

    //Reshape Q into heads;

    callRoPE(k_proj);
    callRoPE(q_proj);

    for(int QHeadIdx = 0; QHeadIdx < Q_HEADS; QHeadIdx++){

        int kHeadIdx = QHeadIdx / GQA_Q_TO_K_RATIO;
        __nv_bfloat16* q_head = q_proj + QHeadIdx * HEAD_DIM;
        __nv_bfloat16* k_head = k_proj + kHeadIdx * HEAD_DIM;
        __nv_bfloat16* v_head = v_proj + kHeadIdx * HEAD_DIM;
        __nv_bfloat16* attentionScore_head = attentionScore + input_token_size * input_token_size * QHeadIdx;
        callmatMul(attentionScore_head, q_head, k_head, input_token_size, HEAD_DIM, input_token_size, CUBLAS_OP_N, CUBLAS_OP_T);
        

    }

        // callmatMul(attentionScore_head, q_head, k_head, input_token_size, HEAD_DIM, input_token_size, 0, 1);




        
      
    



    return 0;
}

int MLP(int layer, __nv_bfloat16* inputScore, __nv_bfloat16* MLPoutput){
    return 0;
}

int main(){
    if(loadWeights()) return 1;

    std::vector<int> input_tokens = {678, 264, 1933, 13};

    int input_token_size = input_tokens.size();

    __nv_bfloat16* embeddedTokens;
    int embedding_size = EMBEDDING_DIMENSIONS * input_token_size * sizeof(__nv_bfloat16);
    if(loadTokensandEmbedding(input_tokens, &embeddedTokens, input_token_size, embedding_size)) return 1;

    /*
    Puting the transformer for loop in code here
    */
    __nv_bfloat16* prevOutput = embeddedTokens;
    for(int layer = 0; layer < N_LAYERS; layer++){

        __nv_bfloat16* rms1;
        if(RMS_layer(prevOutput, &rms1, *weights.input_layer_norm, input_token_size)) return 1;

        __nv_bfloat16* outputScore;
        attentionPass(layer, rms1, input_token_size, embedding_size, outputScore);

        callresidualConnectionsKernel(outputScore, prevOutput, input_token_size, MaxThreadCount);

        __nv_bfloat16* cpydata;
        cudaMemcpy(cpydata, outputScore, input_token_size * input_token_size, cudaMemcpyDeviceToDevice);

        __nv_bfloat16* MLPoutput;
        MLP(layer, outputScore, MLPoutput);

        callresidualConnectionsKernel(MLPoutput, embeddedTokens, input_token_size, MaxThreadCount);

        prevOutput = MLPoutput;

        cout << "Pass " << layer << "Completed successfully\n";
       
    }
    return 0;
}
