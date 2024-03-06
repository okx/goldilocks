#include "cuda_utils.hpp"
#include "cuda_utils.cuh"

#define MAX_GPUS 16

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

void warmup_all_gpus()
{
    uint64_t *gpu_a[MAX_GPUS];
    uint64_t size = (1 << 20);

    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaMalloc(&gpu_a[d], size * sizeof(uint64_t)));
    }
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaFree(gpu_a[d]));
    }
}