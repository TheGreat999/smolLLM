#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map> 
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define JSON_USE_IMPLICIT_CONVERSIONS 0 // this one is useful, but I don't remember why
#include "json.hpp"


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
constexpr int N_LAYERS = 16;
constexpr int EMBEDDING_DIMENSIONS = 2048;
constexpr int VOCABULURY = 128256;

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

int checkGPUStatus(){
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }
    cudaDeviceSynchronize();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B2MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B2MB << "MB, total memory: " << total_mem / B2MB << "MB\n";
    return 0;
}

int cpyToGPU(void* modelWeights, uint64_t maxOffset, std::ifstream& model){
    TIMER_START(Devalloc);
    cudaMalloc(&modelWeights, maxOffset);
    TIMER_END(Devalloc);

    TIMER_START(Hostalloc);
    char* temp_cpu_buffer;
    cudaHostAlloc(&temp_cpu_buffer, maxOffset, cudaHostAllocDefault);
    TIMER_END(Hostalloc);

    TIMER_START(fileRead);
    model.read(temp_cpu_buffer, maxOffset);
    TIMER_END(fileRead);

    TIMER_START(Host2Devcpy);
    cudaError_t err = cudaMemcpy(modelWeights, temp_cpu_buffer, maxOffset, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        cout<<"There was a error when copying weights to the GPU\n";
        cerr<<cudaGetErrorName(err)<<'\n'<<cudaGetErrorString(err);
        model.close();
        return 1;
    }
    TIMER_END(Host2Devcpy);
    cudaFreeHost(temp_cpu_buffer);
    cout<<"Copied "<<maxOffset / B2MB <<" MB of weights to GPU successfully\n";
    model.close();
    return 0;
}

int loadWeights(Weights& weights){

    if(checkGPUStatus() != 0)return 1;
    std::ifstream model("./models/Llama-3.2-1B-Instruct/model.safetensors", std::ios_base::binary);
    if(!model.is_open()){
        cout<<"Could not open model file\n";
        model.close();
        return 1;
    }
    cout<<"Model file opened successfully\n";


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

    void* modelWeights; // This pointer will track the location of the weights;
    // REMEBER TO PASS THE POINTER AS A REFF
    if(cpyToGPU(&modelWeights, maxOffset, model) != 0){
        return 1;
    }

    weights.embed_tokens = (__nv_bfloat16 *)((char *) modelWeights + offsets["model.embed_tokens.weight"]);
    for(int layer = 0; layer < N_LAYERS; layer++){
        weights.input_layer_norm[layer] = (__nv_bfloat16 *) ((char *)maxOffset + offsets["model.layers." +to_string(layer) + ".input_layernorm.weight"]);
        weights.k_proj[layer] = (__nv_bfloat16 *) ((char *)maxOffset + offsets["model.layers." +to_string(layer) + ".self_attn.k_proj.weight"]);
        weights.q_proj[layer] = (__nv_bfloat16 *) ((char *)maxOffset + offsets["model.layers." +to_string(layer) + ".self_attn.q_proj.weight"]);
        weights.v_proj[layer] = (__nv_bfloat16 *) ((char *)maxOffset + offsets["model.layers." +to_string(layer) + ".self_attn.v_proj.weight"]);
        weights.o_proj[layer] = (__nv_bfloat16 *) ((char *)maxOffset + offsets["model.layers." +to_string(layer) + ".self_attn.o_proj.weight"]);
        weights.post_attn_layer_norms[layer] = (__nv_bfloat16 *) ((char *)maxOffset + offsets["model.layers." +to_string(layer) + ".post_attention_layernorm.weight"]);
        weights.mlp_down_proj[layer] = (__nv_bfloat16 *) ((char *)maxOffset + offsets["model.layers." +to_string(layer) + ".mlp.down_proj.weight"]);
        weights.mlp_gate_proj[layer] = (__nv_bfloat16 *) ((char *)maxOffset + offsets["model.layers." +to_string(layer) + ".mlp.gate_proj.weight"]);
        weights.mlp_up_proj[layer] = (__nv_bfloat16 *) ((char *)maxOffset + offsets["model.layers." +to_string(layer) + ".mlp.up_proj.weight"]);
    }
    weights.final_norm = (__nv_bfloat16 *)((char *) modelWeights + offsets["model.norm.weight"]);
    cout<<"Weights pointers marked successfully\n\n";
    checkGPUStatus();
    return 0;

}
int main(){
    Weights weights;
    loadWeights(weights);
    return 0;
}
