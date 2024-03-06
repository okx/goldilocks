#include "cuda_utils.hpp"

void* alloc_pinned_mem(size_t len)
{
    void* ptr;
    cudaError_t status = cudaMallocHost(&ptr, len);
    if (status != cudaSuccess)
    {
        ptr = NULL;
    }
    return ptr;
}

void free_pinned_mem(void* ptr)
{
    cudaFreeHost(ptr);
}