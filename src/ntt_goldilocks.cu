#include "ntt_goldilocks.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "poseidon_goldilocks.hpp"

// CUDA Threads per Block
#define TPB 64
#define MAX_GPUS 16

gl64_t *gpu_roots[16];
gl64_t *gpu_a[16];
gl64_t *gpu_a2[16];
gl64_t *gpu_powTwoInv[16];
gl64_t *gpu_r_[16];
cudaStream_t gpu_stream[16];
gl64_t *state[SPONGE_WIDTH];

#define GPU_TIMING
#ifdef GPU_TIMING
#include <sys/time.h>
struct timeval start;
#endif

__host__ __device__ __forceinline__ u_int64_t BR(u_int64_t x, u_int64_t domainPow)
{
    x = (x >> 16) | (x << 16);
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    return (((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)) >> (32 - domainPow);
}

__device__ __forceinline__ gl64_t root(gl64_t *roots, uint32_t domainPow, uint64_t idx, uint32_t s)
{
    return roots[idx << (s - domainPow)];
}

__host__ __device__ __forceinline__ int intt_idx(int i, int N)
{
    int ind1 = N - i;
    if (ind1 == N)
    {
        ind1 = 0;
    }
    return ind1;
}

__global__ void ntt_iter_loop(uint32_t nBatches, gl64_t *roots, gl64_t *a, gl64_t *a2, gl64_t *powTwoInv, gl64_t *r_,
                              uint64_t batchSize, uint64_t ncols, uint64_t rs, uint64_t re, uint64_t rb, uint64_t rm,
                              uint32_t super_s, uint32_t s, uint32_t sInc, uint32_t maxBatchPow, uint32_t domainPow,
                              uint32_t inverse, uint32_t extend, uint32_t size)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x; // tid
    if (b >= nBatches)
    {
        return;
    }

    for (uint32_t si = 0; si < sInc; si++)
    {
        u_int64_t m = 1 << (s + si);
        u_int64_t mdiv2 = m >> 1;
        u_int64_t mdiv2i = 1 << si;
        u_int64_t mi = mdiv2i * 2;
        for (uint32_t i = 0; i < (batchSize >> 1); i++)
        {
            u_int64_t ki = b * batchSize + (i / mdiv2i) * mi;
            u_int64_t ji = i % mdiv2i;

            u_int64_t offset1 = (ki + ji + mdiv2i) * ncols;
            u_int64_t offset2 = (ki + ji) * ncols;

            u_int64_t j = (b * batchSize / 2 + i);
            j = (j & rm) * rb + (j >> (re - rs));
            j = j % mdiv2;

            gl64_t w = root(roots, s + si, j, super_s);
            for (uint32_t k = 0; k < ncols; ++k)
            {
                gl64_t t = w * a[offset1 + k];
                gl64_t u = a[offset2 + k];

                a[offset2 + k] = t + u;
                a[offset1 + k] = u - t;
            }
        }
    }
    if (s + maxBatchPow <= domainPow || !inverse)
    {
        for (uint32_t x = 0; x < batchSize; x++)
        {
            u_int64_t offset_dstY = (x * nBatches + b) * ncols;
            u_int64_t offset_src = (b * batchSize + x) * ncols;
            std::memcpy(&a2[offset_dstY], &a[offset_src], ncols * sizeof(gl64_t));
        }
    }
    else
    {
        if (extend)
        {
            for (uint32_t x = 0; x < batchSize; x++)
            {
                u_int64_t dsty = intt_idx((x * nBatches + b), size);
                u_int64_t offset_dstY = dsty * ncols;
                u_int64_t offset_src = (b * batchSize + x) * ncols;
                for (uint32_t k = 0; k < ncols; k++)
                {
                    a2[offset_dstY + k] = a[offset_src + k] * r_[dsty];
                }
            }
        }
        else
        {
            for (uint32_t x = 0; x < batchSize; x++)
            {
                u_int64_t dsty = intt_idx((x * nBatches + b), size);
                u_int64_t offset_dstY = dsty * ncols;
                u_int64_t offset_src = (b * batchSize + x) * ncols;
                for (uint64_t k = 0; k < ncols; k++)
                {
                    a2[offset_dstY + k] = a[offset_src + k] * powTwoInv[domainPow];
                }
            }
        }
    }
}

/**
 * @brief permutation of components of an array in bit-reversal order. If dst==src the permutation is performed on-site.
 *
 * @param dst destination pointer (may be equal to src)
 * @param src source pointer
 * @param size field size
 * @param offset_cols columns offset (for NTT wifh nblock>1)
 * @param ncols number of columns of destination array
 * @param ncols_all number of columns of source array (ncols = nocols_all if nblock == 1)
 */
void NTT_Goldilocks::reversePermutation(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all)
{
    uint32_t domainSize = log2(size);
    if (dst != src)
    {
        if (extension <= 1)
        {
#pragma omp parallel for schedule(static)
            for (u_int64_t i = 0; i < size; i++)
            {
                u_int64_t r = BR(i, domainSize);
                u_int64_t offset_r1 = r * ncols_all + offset_cols;
                u_int64_t offset_i1 = i * ncols;
                std::memcpy(&dst[offset_i1], &src[offset_r1], ncols * sizeof(Goldilocks::Element));
            }
        }
        else
        {
            u_int64_t ext_ = (size / extension) * ncols_all;

#pragma omp parallel for schedule(static)
            for (u_int64_t i = 0; i < size; i++)
            {
                u_int64_t r = BR(i, domainSize);
                u_int64_t offset_r1 = r * ncols_all + offset_cols;
                u_int64_t offset_i1 = i * ncols;
                if (offset_r1 < ext_)
                {
                    std ::memcpy(&dst[offset_i1], &src[offset_r1], ncols * sizeof(Goldilocks::Element));
                }
                else
                {
                    std::memset(&dst[offset_i1], 0, ncols * sizeof(Goldilocks::Element));
                }
            }
        }
    }
    else
    {
        if (extension <= 1)
        {
            assert(offset_cols == 0 && ncols == ncols_all); // single block
#pragma omp parallel for schedule(static)
            for (u_int64_t i = 0; i < size; i++)
            {
                u_int64_t r = BR(i, domainSize);
                u_int64_t offset_r = r * ncols;
                u_int64_t offset_i = i * ncols;
                if (r < i)
                {
                    Goldilocks::Element tmp[ncols];
                    std::memcpy(&tmp[0], &src[offset_r], ncols * sizeof(Goldilocks::Element));
                    std::memcpy(&dst[offset_r], &src[offset_i], ncols * sizeof(Goldilocks::Element));
                    std::memcpy(&dst[offset_i], &tmp[0], ncols * sizeof(Goldilocks::Element));
                }
            }
        }
        else
        {
            assert(0); // Option not implemented yet
        }
    }
}

__global__ void reversePermutationGPU(uint64_t *dst, uint64_t *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, int extension, uint32_t domainSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid
    if (tid >= size)
    {
        return;
    }

    if (dst != src)
    {
        if (extension <= 1)
        {

            u_int64_t r = BR(tid, domainSize);
            u_int64_t offset_r1 = r * ncols_all + offset_cols;
            u_int64_t offset_i1 = tid * ncols;
            std::memcpy(&dst[offset_i1], &src[offset_r1], ncols * sizeof(Goldilocks::Element));
        }
        else
        {
            u_int64_t ext_ = (size / extension) * ncols_all;

            u_int64_t r = BR(tid, domainSize);
            u_int64_t offset_r1 = r * ncols_all + offset_cols;
            u_int64_t offset_i1 = tid * ncols;
            if (offset_r1 < ext_)
            {
                std ::memcpy(&dst[offset_i1], &src[offset_r1], ncols * sizeof(Goldilocks::Element));
            }
            else
            {
                std::memset(&dst[offset_i1], 0, ncols * sizeof(Goldilocks::Element));
            }
        }
    }
    else
    {
        if (extension <= 1)
        {
            assert(offset_cols == 0 && ncols == ncols_all); // single block

            u_int64_t r = BR(tid, domainSize);
            u_int64_t offset_r = r * ncols;
            u_int64_t offset_i = tid * ncols;
            if (r < tid)
            {
                auto tmp = new Goldilocks::Element[1024];
                std::memcpy(&tmp[0], &src[offset_r], ncols * sizeof(Goldilocks::Element));
                std::memcpy(&dst[offset_r], &src[offset_i], ncols * sizeof(Goldilocks::Element));
                std::memcpy(&dst[offset_i], &tmp[0], ncols * sizeof(Goldilocks::Element));
                delete tmp;
            }
        }
        else
        {
            assert(0); // Option not implemented yet
        }
    }
}

void NTT_Goldilocks::NTT_iters(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, Goldilocks::Element *aux, bool inverse, bool extend)
{
    Goldilocks::Element *dst_;
    if (dst != NULL)
    {
        dst_ = dst;
    }
    else
    {
        dst_ = src;
    }
    Goldilocks::Element *a = dst_;
    Goldilocks::Element *a2 = aux;
    Goldilocks::Element *tmp;

    u_int64_t domainPow = log2(size);
    assert(((u_int64_t)1 << domainPow) == size);
    if (nphase < 1 || domainPow == 0)
    {
        nphase = 1;
    }
    else if (nphase > domainPow)
    {
        nphase = domainPow;
    }
    u_int64_t maxBatchPow = s / nphase;
    u_int64_t res = s % nphase;
    if (res > 0)
    {
        maxBatchPow += 1;
    }
    bool iseven = true;
    tmp = a;
    if (nphase % 2 == 1)
    {
        iseven = false;
        tmp = a2;
    }
    reversePermutation(tmp, src, size, offset_cols, ncols, ncols_all);
    if (iseven == false)
    {
        tmp = a2;
        a2 = a;
        a = tmp;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(nThreads);
    uint64_t count = 1;
    for (u_int64_t s = 1; s <= domainPow; s += maxBatchPow, ++count)
    {
        if (res > 0 && count == res + 1 && maxBatchPow > 1)
        {
            maxBatchPow -= 1;
        }
        u_int64_t sInc = s + maxBatchPow <= domainPow ? maxBatchPow : domainPow - s + 1;
        u_int64_t rs = s - 1;
        u_int64_t re = domainPow - 1;
        u_int64_t rb = 1 << rs;
        u_int64_t rm = (1 << (re - rs)) - 1;
        u_int64_t batchSize = 1 << sInc;
        u_int64_t nBatches = size / batchSize;

        int chunk1 = nBatches / nThreads;
        if (chunk1 == 0)
        {
            chunk1 = 1;
        }

#pragma omp parallel for schedule(static, chunk1)
        for (u_int64_t b = 0; b < nBatches; b++)
        {
            for (u_int64_t si = 0; si < sInc; si++)
            {
                u_int64_t m = 1 << (s + si);
                u_int64_t mdiv2 = m >> 1;
                u_int64_t mdiv2i = 1 << si;
                u_int64_t mi = mdiv2i * 2;
                for (u_int64_t i = 0; i < (batchSize >> 1); i++)
                {
                    u_int64_t ki = b * batchSize + (i / mdiv2i) * mi;
                    u_int64_t ji = i % mdiv2i;

                    u_int64_t offset1 = (ki + ji + mdiv2i) * ncols;
                    u_int64_t offset2 = (ki + ji) * ncols;

                    u_int64_t j = (b * batchSize / 2 + i);
                    j = (j & rm) * rb + (j >> (re - rs));
                    j = j % mdiv2;

                    Goldilocks::Element w = root(s + si, j);
                    for (u_int64_t k = 0; k < ncols; ++k)
                    {
                        Goldilocks::Element t = w * a[offset1 + k];
                        Goldilocks::Element u = a[offset2 + k];

                        Goldilocks::add(a[offset2 + k], t, u);
                        Goldilocks::sub(a[offset1 + k], u, t);
                    }
                }
            }
            if (s + maxBatchPow <= domainPow || !inverse)
            {
                for (u_int64_t x = 0; x < batchSize; x++)
                {
                    u_int64_t offset_dstY = (x * nBatches + b) * ncols;
                    u_int64_t offset_src = (b * batchSize + x) * ncols;
                    std::memcpy(&a2[offset_dstY], &a[offset_src], ncols * sizeof(Goldilocks::Element));
                }
            }
            else
            {
                if (extend)
                {
                    for (u_int64_t x = 0; x < batchSize; x++)
                    {
                        u_int64_t dsty = intt_idx((x * nBatches + b), size);
                        u_int64_t offset_dstY = dsty * ncols;
                        u_int64_t offset_src = (b * batchSize + x) * ncols;
                        for (uint64_t k = 0; k < ncols; k++)
                        {
                            Goldilocks::mul(a2[offset_dstY + k], a[offset_src + k], r_[dsty]);
                        }
                    }
                }
                else
                {
                    for (u_int64_t x = 0; x < batchSize; x++)
                    {
                        u_int64_t dsty = intt_idx((x * nBatches + b), size);
                        u_int64_t offset_dstY = dsty * ncols;
                        u_int64_t offset_src = (b * batchSize + x) * ncols;
                        for (uint64_t k = 0; k < ncols; k++)
                        {
                            Goldilocks::mul(a2[offset_dstY + k], a[offset_src + k], powTwoInv[domainPow]);
                        }
                    }
                }
            }
        }
        tmp = a2;
        a2 = a;
        a = tmp;
    }
    if (a != dst_)
    {
        if (size > 1)
        {
            assert(0); // should never need this copy...
        }
        Goldilocks::parcpy(dst_, a, size * ncols, nThreads);
    }
}

/*
 * Only 1 GPU
 */
void NTT_Goldilocks::NTT_GPU_iters(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, Goldilocks::Element *aux, bool inverse, bool extend, uint64_t aux_size, bool copyToGPU, bool copyFromGPU)
{
    Goldilocks::Element *dst_;
    if (dst != NULL)
    {
        dst_ = dst;
    }
    else
    {
        dst_ = src;
    }
    Goldilocks::Element *a = dst_;
    Goldilocks::Element *a2 = aux;
    Goldilocks::Element *tmp;

    int gpu_id = 0;

    u_int64_t domainPow = log2(size);
    assert(((u_int64_t)1 << domainPow) == size);
    if (nphase < 1 || domainPow == 0)
    {
        nphase = 1;
    }
    else if (nphase > domainPow)
    {
        nphase = domainPow;
    }
    u_int64_t maxBatchPow = s / nphase;
    u_int64_t res = s % nphase;
    if (res > 0)
    {
        maxBatchPow += 1;
    }
    bool iseven = true;
    tmp = a;
    if (nphase % 2 == 1)
    {
        iseven = false;
        tmp = a2;
    }
    reversePermutation(tmp, src, size, offset_cols, ncols, ncols_all);
    if (iseven == false)
    {
        tmp = a2;
        a2 = a;
        a = tmp;
    }

    uint64_t actual_size = aux_size;
    if (copyToGPU)
    {
        CHECKCUDAERR(cudaSetDevice(gpu_id));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_a[gpu_id], (uint64_t *)a, actual_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_a2[gpu_id], (uint64_t *)a2, actual_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    }

    uint64_t count = 1;
    for (u_int64_t s = 1; s <= domainPow; s += maxBatchPow, ++count)
    {
        if (res > 0 && count == res + 1 && maxBatchPow > 1)
        {
            maxBatchPow -= 1;
        }
        u_int64_t sInc = s + maxBatchPow <= domainPow ? maxBatchPow : domainPow - s + 1;
        u_int64_t rs = s - 1;
        u_int64_t re = domainPow - 1;
        u_int64_t rb = 1 << rs;
        u_int64_t rm = (1 << (re - rs)) - 1;
        u_int64_t batchSize = 1 << sInc;
        u_int64_t nBatches = size / batchSize;

        ntt_iter_loop<<<nBatches / TPB + 1, TPB, 0, gpu_stream[gpu_id]>>>(nBatches, gpu_roots[gpu_id], gpu_a[gpu_id], gpu_a2[gpu_id], gpu_powTwoInv[gpu_id], gpu_r_[gpu_id], batchSize, ncols, rs, re, rb, rm, this->s, s, sInc, maxBatchPow, domainPow, inverse, extend, size);
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));

        gl64_t *tmpg = gpu_a2[gpu_id];
        gpu_a2[gpu_id] = gpu_a[gpu_id];
        gpu_a[gpu_id] = tmpg;
    }

    if (a != dst_)
    {
        a = dst_;
    }

    if (copyFromGPU)
    {
        CHECKCUDAERR(cudaMemcpyAsync((uint64_t *)a, gpu_a[gpu_id], actual_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[gpu_id]));
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
    }
}

void NTT_Goldilocks::NTT_GPU_iters_onGPU(Goldilocks::Element *dst, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, Goldilocks::Element *aux, bool inverse, bool extend, uint64_t aux_size, bool copyFromGPU)
{
    int gpu_id = 0;
    gl64_t *a = gpu_a[gpu_id];
    gl64_t *a2 = gpu_a2[gpu_id];
    gl64_t *tmp;

    u_int64_t domainPow = log2(size);
    assert(((u_int64_t)1 << domainPow) == size);
    if (nphase < 1 || domainPow == 0)
    {
        nphase = 1;
    }
    else if (nphase > domainPow)
    {
        nphase = domainPow;
    }
    u_int64_t maxBatchPow = s / nphase;
    u_int64_t res = s % nphase;
    if (res > 0)
    {
        maxBatchPow += 1;
    }
    bool iseven = true;
    tmp = a;
    if (nphase % 2 == 1)
    {
        iseven = false;
        tmp = a2;
    }
    reversePermutationGPU<<<size / TPB + 1, TPB>>>((uint64_t *)tmp, (uint64_t *)gpu_a[gpu_id], size, offset_cols, ncols, ncols_all, extension, log2(size));
    if (iseven == false)
    {
        tmp = a2;
        a2 = a;
        a = tmp;
    }

    uint64_t actual_size = aux_size;

    uint64_t count = 1;
    for (u_int64_t s = 1; s <= domainPow; s += maxBatchPow, ++count)
    {
        if (res > 0 && count == res + 1 && maxBatchPow > 1)
        {
            maxBatchPow -= 1;
        }
        u_int64_t sInc = s + maxBatchPow <= domainPow ? maxBatchPow : domainPow - s + 1;
        u_int64_t rs = s - 1;
        u_int64_t re = domainPow - 1;
        u_int64_t rb = 1 << rs;
        u_int64_t rm = (1 << (re - rs)) - 1;
        u_int64_t batchSize = 1 << sInc;
        u_int64_t nBatches = size / batchSize;

        ntt_iter_loop<<<nBatches / TPB + 1, TPB, 0, gpu_stream[gpu_id]>>>(nBatches, gpu_roots[gpu_id], a, a2, gpu_powTwoInv[gpu_id], gpu_r_[gpu_id], batchSize, ncols, rs, re, rb, rm, this->s, s, sInc, maxBatchPow, domainPow, inverse, extend, size);
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));

        tmp = a2;
        a2 = a;
        a = tmp;
    }

    if (copyFromGPU)
    {
        CHECKCUDAERR(cudaMemcpyAsync((uint64_t *)dst, (uint64_t *)a, actual_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[gpu_id]));
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
    }
}

void NTT_Goldilocks::NTT_MultiGPU_iters(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, Goldilocks::Element *aux, bool inverse, bool extend, uint64_t aux_size, uint64_t aux_size_last, int ngpus, int gpu_id)
{
    Goldilocks::Element *dst_;
    if (dst != NULL)
    {
        dst_ = dst;
    }
    else
    {
        dst_ = src;
    }
    Goldilocks::Element *a = dst_;
    Goldilocks::Element *a2 = aux;
    Goldilocks::Element *tmp;

    u_int64_t domainPow = log2(size);
    assert(((u_int64_t)1 << domainPow) == size);
    if (nphase < 1 || domainPow == 0)
    {
        nphase = 1;
    }
    else if (nphase > domainPow)
    {
        nphase = domainPow;
    }
    u_int64_t maxBatchPow = s / nphase;
    u_int64_t res = s % nphase;
    if (res > 0)
    {
        maxBatchPow += 1;
    }
    bool iseven = true;
    tmp = a;
    if (nphase % 2 == 1)
    {
        iseven = false;
        tmp = a2;
    }
    reversePermutation(tmp, src, size, offset_cols, ncols, ncols_all);
    if (iseven == false)
    {
        tmp = a2;
        a2 = a;
        a = tmp;
    }

    CHECKCUDAERR(cudaSetDevice(gpu_id));
    uint64_t actual_size = (gpu_id == ngpus - 1) ? aux_size_last : aux_size;
    CHECKCUDAERR(cudaMemcpyAsync(gpu_a[gpu_id], (uint64_t *)a, actual_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_a2[gpu_id], (uint64_t *)a2, actual_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));

#ifdef GPU_TIMING
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
    struct timeval end;
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("NTT_MultiGPU_iters: copy data to GPU %d took: %lu ms\n", gpu_id, t / 1000);
#endif

    uint64_t count = 1;
    for (u_int64_t s = 1; s <= domainPow; s += maxBatchPow, ++count)
    {
        if (res > 0 && count == res + 1 && maxBatchPow > 1)
        {
            maxBatchPow -= 1;
        }
        u_int64_t sInc = s + maxBatchPow <= domainPow ? maxBatchPow : domainPow - s + 1;
        u_int64_t rs = s - 1;
        u_int64_t re = domainPow - 1;
        u_int64_t rb = 1 << rs;
        u_int64_t rm = (1 << (re - rs)) - 1;
        u_int64_t batchSize = 1 << sInc;
        u_int64_t nBatches = size / batchSize;

        ntt_iter_loop<<<nBatches / TPB + 1, TPB, 0, gpu_stream[gpu_id]>>>(nBatches, gpu_roots[gpu_id], gpu_a[gpu_id], gpu_a2[gpu_id], gpu_powTwoInv[gpu_id], gpu_r_[gpu_id], batchSize, ncols, rs, re, rb, rm, this->s, s, sInc, maxBatchPow, domainPow, inverse, extend, size);
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));

        gl64_t *tmpg = gpu_a2[gpu_id];
        gpu_a2[gpu_id] = gpu_a[gpu_id];
        gpu_a[gpu_id] = tmpg;
    }

#ifdef GPU_TIMING
    struct timeval end2;
    gettimeofday(&end2, NULL);
    t = end2.tv_sec * 1000000 + end2.tv_usec - end.tv_sec * 1000000 - end.tv_usec;
    printf("NTT_MultiGPU_iters: kernel on GPU %d took: %lu ms\n", gpu_id, t / 1000);
#endif

    if (a != dst_)
    {
        a = dst_;
    }

    CHECKCUDAERR(cudaMemcpyAsync((uint64_t *)a, gpu_a[gpu_id], actual_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));

#ifdef GPU_TIMING
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - end2.tv_sec * 1000000 - end2.tv_usec;
    printf("NTT_MultiGPU_iters: copy data from GPU %d took: %lu ms\n", gpu_id, t / 1000);
#endif
}

void NTT_Goldilocks::NTT(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, u_int64_t nblock, bool inverse, bool extend)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }
    if (nblock < 1)
    {
        nblock = 1;
    }
    if (nblock > ncols)
    {
        nblock = ncols;
    }

    u_int64_t offset_cols = 0;
    u_int64_t ncols_block = ncols / nblock;
    u_int64_t ncols_res = ncols % nblock;
    u_int64_t ncols_alloc = ncols_block;
    if (ncols_res > 0)
    {
        ncols_alloc += 1;
    }
    Goldilocks::Element *dst_ = NULL;
    Goldilocks::Element *aux = NULL;
    if (buffer == NULL)
    {
        aux = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * size * ncols_alloc);
    }
    else
    {
        aux = buffer;
    }
    if (nblock > 1)
    {
        dst_ = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * size * ncols_alloc);
    }
    else
    {
        dst_ = dst;
    }
    for (u_int64_t ib = 0; ib < nblock; ++ib)
    {
        uint64_t aux_ncols = ncols_block;
        if (ib < ncols_res)
            aux_ncols += 1;
        NTT_Goldilocks::NTT_iters(dst_, src, size, offset_cols, aux_ncols, ncols, nphase, aux, inverse, extend);
        if (nblock > 1)
        {
#pragma omp parallel for schedule(static)
            for (u_int64_t ie = 0; ie < size; ++ie)
            {
                u_int64_t offset2 = ie * ncols + offset_cols;
                std::memcpy(&dst[offset2], &dst_[ie * aux_ncols], aux_ncols * sizeof(Goldilocks::Element));
            }
        }
        offset_cols += aux_ncols;
    }
    if (nblock > 1)
    {
        free(dst_);
    }
    if (buffer == NULL)
    {
        free(aux);
    }
}

/**
 * 1 GPU
 */
void NTT_Goldilocks::NTT_GPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, bool inverse, bool extend)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    Goldilocks::Element *aux = NULL;
    uint64_t aux_size = size * ncols;
    if (buffer == NULL)
    {
        aux = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);
    }
    else
    {
        aux = buffer;
    }

#ifdef GPU_TIMING
    struct timeval start;
    gettimeofday(&start, NULL);
#endif

    int gpu_id = 0;
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaStreamCreate(gpu_stream + gpu_id));
    CHECKCUDAERR(cudaMalloc(&gpu_roots[gpu_id], nRoots * sizeof(uint64_t)));      // 64M
    CHECKCUDAERR(cudaMalloc(&gpu_powTwoInv[gpu_id], (s + 1) * sizeof(uint64_t))); // small
    if (extend)
    {
        CHECKCUDAERR(cudaMalloc(&gpu_r_[gpu_id], size * sizeof(uint64_t))); // 64M
        CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[gpu_id], (uint64_t *)r_, size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    }
    CHECKCUDAERR(cudaMalloc(&gpu_a[gpu_id], aux_size * sizeof(uint64_t)));  // 42G
    CHECKCUDAERR(cudaMalloc(&gpu_a2[gpu_id], aux_size * sizeof(uint64_t))); // 42G
    CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[gpu_id], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_powTwoInv[gpu_id], (uint64_t *)powTwoInv, (s + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));

#ifdef GPU_TIMING
    struct timeval end;
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("NTT_GPU: CPU -> GPU memcpy took: %lu ms\n", t / 1000);
    gettimeofday(&start, NULL);
#endif

    NTT_Goldilocks::NTT_GPU_iters(dst, src, size, 0, ncols, ncols, nphase, aux, inverse, extend, aux_size);

#ifdef GPU_TIMING
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("NTT_GPU: Kernel took: %lu ms\n", t / 1000);
#endif

    if (buffer == NULL)
    {
        free(aux);
    }

    CHECKCUDAERR(cudaStreamDestroy(gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_roots[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a2[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_powTwoInv[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_r_[gpu_id]));
}

void NTT_Goldilocks::NTT_MultiGPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, bool inverse, bool extend)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    uint64_t ncols_per_gpu = (ncols % nDevices) ? ncols / nDevices + 1 : ncols / nDevices;
    uint64_t ncols_last_gpu = ncols - ncols_per_gpu * (nDevices - 1);
    uint64_t aux_size = size * ncols_per_gpu;
    uint64_t aux_size_last = size * ncols_last_gpu;

    printf("Number of GPUs: %d\n", nDevices);
    printf("Number columns: %lu\n", ncols);
    printf("Cols per GPU: %lu\n", ncols_per_gpu);
    printf("Cols last GPU: %lu\n", ncols_last_gpu);
    // TODO - we suppose the GPU memory is large enough

    Goldilocks::Element *aux[MAX_GPUS];
    Goldilocks::Element *dst_[MAX_GPUS];
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        aux[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * size * ncols_per_gpu);
        dst_[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * size * ncols_per_gpu);
    }

#ifdef GPU_TIMING
    // global start
    gettimeofday(&start, NULL);
#endif

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaMalloc(&gpu_roots[d], nRoots * sizeof(uint64_t)));      // 64M
        CHECKCUDAERR(cudaMalloc(&gpu_powTwoInv[d], (s + 1) * sizeof(uint64_t))); // small
        if (extend)
        {
            CHECKCUDAERR(cudaMalloc(&gpu_r_[d], size * sizeof(uint64_t))); // 64M
            CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[d], (uint64_t *)r_, size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        }
        CHECKCUDAERR(cudaMalloc(&gpu_a[d], aux_size * sizeof(uint64_t)));  // 42G
        CHECKCUDAERR(cudaMalloc(&gpu_a2[d], aux_size * sizeof(uint64_t))); // 42G
        CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[d], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_powTwoInv[d], (uint64_t *)powTwoInv, (s + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
    }

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        NTT_Goldilocks::NTT_MultiGPU_iters(dst_[d], src, size, d * ncols_per_gpu, aux_ncols, ncols, nphase, aux[d], inverse, extend, aux_size, aux_size_last, nDevices, d);
    }

#ifdef GPU_TIMING
    printf("NTT_MultiGPU: copy back ...\n");
    gettimeofday(&start, NULL);
#endif

    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
#pragma omp parallel for schedule(static)
        for (u_int64_t ie = 0; ie < size; ++ie)
        {
            u_int64_t offset2 = ie * ncols + d * ncols_per_gpu;
            std::memcpy(&dst[offset2], &(dst_[d][ie * aux_ncols]), aux_ncols * sizeof(Goldilocks::Element));
        }
    }

#ifdef GPU_TIMING
    struct timeval end;
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("NTT_MultiGPU: CPU memcpy took: %lu ms\n", t / 1000);
#endif

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
        CHECKCUDAERR(cudaFree(gpu_roots[d]));
        CHECKCUDAERR(cudaFree(gpu_a[d]));
        CHECKCUDAERR(cudaFree(gpu_a2[d]));
        CHECKCUDAERR(cudaFree(gpu_powTwoInv[d]));
        CHECKCUDAERR(cudaFree(gpu_r_[d]));
        free(aux[d]);
        free(dst_[d]);
    }
}

void NTT_Goldilocks::NTT_BatchGPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, u_int64_t ncols_batch, Goldilocks::Element *buffer, u_int64_t nphase, bool inverse, bool extend)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    uint64_t nbatches = ncols / ncols_batch;
    uint64_t ncols_last_batch = ncols - ncols_batch * nbatches;
    if (ncols_last_batch > 0)
    {
        nbatches++;
    }
    uint64_t aux_size = size * ncols_batch;

    printf("Number of columns: %lu\n", ncols);
    printf("Number of batches: %lu\n", nbatches);
    printf("Cols per batch: %lu\n", ncols_batch);
    printf("Cols last batch: %lu\n", ncols_last_batch);

    Goldilocks::Element *aux;
    Goldilocks::Element *dst_;
    aux = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);
    dst_ = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);

#ifdef GPU_TIMING
    // global start
    gettimeofday(&start, NULL);
#endif

    int gpu_id = 0;
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaStreamCreate(gpu_stream + gpu_id));
    CHECKCUDAERR(cudaMalloc(&gpu_roots[gpu_id], nRoots * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_powTwoInv[gpu_id], (s + 1) * sizeof(uint64_t)));
    if (extend)
    {
        CHECKCUDAERR(cudaMalloc(&gpu_r_[gpu_id], size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[gpu_id], (uint64_t *)r_, size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    }
    CHECKCUDAERR(cudaMalloc(&gpu_a[gpu_id], aux_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_a2[gpu_id], aux_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[gpu_id], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_powTwoInv[gpu_id], (uint64_t *)powTwoInv, (s + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));

    for (int b = 0; b < nbatches; b++)
    {
        uint64_t aux_ncols = (b == nbatches - 1 && ncols_last_batch > 0) ? ncols_last_batch : ncols_batch;
        NTT_Goldilocks::NTT_GPU_iters(dst_, src, size, b * ncols_batch, aux_ncols, ncols, nphase, aux, inverse, extend, aux_size, true, true);

#pragma omp parallel for schedule(static)
        for (u_int64_t ie = 0; ie < size; ++ie)
        {
            u_int64_t offset2 = ie * ncols + b * ncols_batch;
            std::memcpy(&dst[offset2], &(dst_[ie * aux_ncols]), aux_ncols * sizeof(Goldilocks::Element));
        }
    }

#ifdef GPU_TIMING
    struct timeval end;
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("NTT_BatchGPU: CPU memcpy took: %lu ms\n", t / 1000);
#endif

    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaStreamDestroy(gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_roots[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a2[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_powTwoInv[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_r_[gpu_id]));
    free(aux);
    free(dst_);
}

void NTT_Goldilocks::INTT(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, u_int64_t nblock, bool extend)
{

    if (ncols == 0 || size == 0)
    {
        return;
    }
    Goldilocks::Element *dst_;
    if (dst == NULL)
    {
        dst_ = src;
    }
    else
    {
        dst_ = dst;
    }
    NTT(dst_, src, size, ncols, buffer, nphase, nblock, true, extend);
}

void NTT_Goldilocks::INTT_GPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, bool extend)
{

    if (ncols == 0 || size == 0)
    {
        return;
    }
    Goldilocks::Element *dst_;
    if (dst == NULL)
    {
        dst_ = src;
    }
    else
    {
        dst_ = dst;
    }
    NTT_GPU(dst_, src, size, ncols, buffer, nphase, true, extend);
}

void NTT_Goldilocks::INTT_MultiGPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, bool extend)
{

    if (ncols == 0 || size == 0)
    {
        return;
    }
    Goldilocks::Element *dst_;
    if (dst == NULL)
    {
        dst_ = src;
    }
    else
    {
        dst_ = dst;
    }
    NTT_MultiGPU(dst_, src, size, ncols, buffer, nphase, true, extend);
}

void NTT_Goldilocks::extendPol(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t N_Extended, uint64_t N, uint64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, u_int64_t nblock)
{
    NTT_Goldilocks ntt_extension(N_Extended, nThreads, N_Extended / N);

    Goldilocks::Element *tmp = NULL;
    if (buffer == NULL)
    {
        tmp = (Goldilocks::Element *)malloc(N_Extended * ncols * sizeof(Goldilocks::Element));
    }
    else
    {
        tmp = buffer;
    }
    // TODO: Pre-compute r
    if (r == NULL)
    {
        computeR(N);
    }

    INTT(output, input, N, ncols, tmp, nphase, nblock, true);
    ntt_extension.NTT(output, output, N_Extended, ncols, tmp, nphase, nblock);

    if (buffer == NULL)
    {
        free(tmp);
    }
}

void NTT_Goldilocks::LDE_MerkleTree_CPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, Goldilocks::Element *buffer, bool buildMerkleTree, u_int64_t nphase)
{
    if (buildMerkleTree)
    {
        uint64_t aux_size = ext_size * ncols;
        Goldilocks::Element *dst_ = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);
        extendPol(dst_, src, ext_size, size, ncols, buffer, nphase, 1);
        PoseidonGoldilocks::merkletree_avx(dst, dst_, ncols, ext_size);
        free(dst_);
    }
    else
    {
        extendPol(dst, src, ext_size, size, ncols, buffer, nphase, 1);
    }
}

void NTT_Goldilocks::LDE_MerkleTree_GPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, Goldilocks::Element *buffer, bool buildMerkleTree, u_int64_t nphase)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    if (r == NULL)
    {
        computeR(size);
    }

    int gpu_id = 0;

    uint64_t aux_size = ext_size * ncols;
    Goldilocks::Element *aux = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);
    Goldilocks::Element *dst_ = dst;
    if (buildMerkleTree)
    {
        dst_ = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);
    }
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaStreamCreate(&gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaMalloc(&gpu_roots[gpu_id], nRoots * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_powTwoInv[gpu_id], (s + 1) * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_r_[gpu_id], size * sizeof(uint64_t))); // 64M
    CHECKCUDAERR(cudaMalloc(&gpu_a[gpu_id], aux_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_a2[gpu_id], aux_size * sizeof(uint64_t)));

#ifdef GPU_TIMING
    struct timeval start;
    gettimeofday(&start, NULL);
#endif

    CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[gpu_id], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_powTwoInv[gpu_id], (uint64_t *)powTwoInv, (s + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[gpu_id], (uint64_t *)r_, size * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaDeviceSynchronize());

#ifdef GPU_TIMING
    struct timeval end;
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend_GPU: CPU -> GPU data transfer took: %lu ms\n", t / 1000);
    gettimeofday(&start, NULL);
#endif

    // INTT with extension (inverse and extend set to true), copyToGPU set to true, copyFromGPU set to false
    NTT_Goldilocks::NTT_GPU_iters(dst_, src, size, 0, ncols, ncols, nphase, aux, true, true, aux_size, true, false);

#ifdef GPU_TIMING
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend_GPU: INTT took: %lu ms\n", t / 1000);
    gettimeofday(&start, NULL);
#endif

    // NTT on the extended buffers, copyToGPU set to false, copyFromGPU set to false
    NTT_Goldilocks::NTT_GPU_iters_onGPU(dst_, ext_size, 0, ncols, ncols, nphase, aux, false, false, aux_size, !buildMerkleTree);

#ifdef GPU_TIMING
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend_GPU: NTT took: %lu ms\n", t / 1000);
    gettimeofday(&start, NULL);
#endif

    if (buildMerkleTree)
    {
        PoseidonGoldilocks::merkletree_cuda_gpudata(dst, (uint64_t *)gpu_a[gpu_id], ncols, ext_size);
    }

#ifdef GPU_TIMING
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend_GPU: Merkle tree took: %lu ms\n", t / 1000);
    gettimeofday(&start, NULL);
#endif

    CHECKCUDAERR(cudaStreamDestroy(gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_roots[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a2[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_powTwoInv[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_r_[gpu_id]));
    free(aux);
    if (buildMerkleTree)
    {
        free(dst_);
    }
}

void NTT_Goldilocks::Extend_MultiGPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    uint64_t ncols_per_gpu = (ncols % nDevices) ? ncols / nDevices + 1 : ncols / nDevices;
    uint64_t ncols_last_gpu = ncols - ncols_per_gpu * (nDevices - 1);
    uint64_t aux_size = size * ncols_per_gpu;
    uint64_t aux_size_last = size * ncols_last_gpu;
    uint64_t aux_ext_size = ext_size * ncols_per_gpu;
    uint64_t aux_ext_size_last = ext_size * ncols_last_gpu;

    printf("Number of GPUs: %d\n", nDevices);
    printf("Number columns: %lu\n", ncols);
    printf("Cols per GPU: %lu\n", ncols_per_gpu);
    printf("Cols last GPU: %lu\n", ncols_last_gpu);
    // TODO - we suppose the GPU memory is large enough

    Goldilocks::Element *aux[MAX_GPUS];
    Goldilocks::Element *dst_[MAX_GPUS];
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        aux[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * ext_size * ncols_per_gpu);
        dst_[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * ext_size * ncols_per_gpu);
    }

#ifdef GPU_TIMING
    gettimeofday(&start, NULL);
#endif

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaMalloc(&gpu_roots[d], nRoots * sizeof(uint64_t)));      // 64M
        CHECKCUDAERR(cudaMalloc(&gpu_powTwoInv[d], (s + 1) * sizeof(uint64_t))); // small
        CHECKCUDAERR(cudaMalloc(&gpu_r_[d], ext_size * sizeof(uint64_t)));       // 64M
        CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[d], (uint64_t *)r_, ext_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMalloc(&gpu_a[d], aux_ext_size * sizeof(uint64_t)));  // 42G
        CHECKCUDAERR(cudaMalloc(&gpu_a2[d], aux_ext_size * sizeof(uint64_t))); // 42G
        CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[d], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_powTwoInv[d], (uint64_t *)powTwoInv, (s + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
    }

    // INTT with extension (inverse and extend set to true)
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        NTT_Goldilocks::NTT_MultiGPU_iters(dst_[d], src, size, d * ncols_per_gpu, aux_ncols, ncols, nphase, aux[d], true, true, aux_size, aux_size_last, nDevices, d);
    }

#ifdef GPU_TIMING
    printf("Extend_MultiGPU: copy back after INTT ...\n");
    gettimeofday(&start, NULL);
#endif

    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
#pragma omp parallel for schedule(static)
        for (u_int64_t ie = 0; ie < size; ++ie)
        {
            u_int64_t offset2 = ie * ncols + d * ncols_per_gpu;
            std::memcpy(&dst[offset2], &(dst_[d][ie * aux_ncols]), aux_ncols * sizeof(Goldilocks::Element));
        }
    }
#ifdef GPU_TIMING
    struct timeval end;
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend_MultiGPU: CPU memcpy took: %lu ms\n", t / 1000);
#endif

    // NTT on the extended domain (inverse and extend set to false)
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        NTT_Goldilocks::NTT_MultiGPU_iters(dst_[d], dst, ext_size, d * ncols_per_gpu, aux_ncols, ncols, nphase, aux[d], false, false, aux_ext_size, aux_ext_size_last, nDevices, d);
    }

#ifdef GPU_TIMING
    printf("Extend_MultiGPU: copy back after NTT...\n");
    gettimeofday(&start, NULL);
#endif
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
#pragma omp parallel for schedule(static)
        for (u_int64_t ie = 0; ie < ext_size; ++ie)
        {
            u_int64_t offset2 = ie * ncols + d * ncols_per_gpu;
            std::memcpy(&dst[offset2], &(dst_[d][ie * aux_ncols]), aux_ncols * sizeof(Goldilocks::Element));
        }
    }
#ifdef GPU_TIMING
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend_MultiGPU: CPU memcpy took: %lu ms\n", t / 1000);
#endif

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
        CHECKCUDAERR(cudaFree(gpu_roots[d]));
        CHECKCUDAERR(cudaFree(gpu_a[d]));
        CHECKCUDAERR(cudaFree(gpu_a2[d]));
        CHECKCUDAERR(cudaFree(gpu_powTwoInv[d]));
        CHECKCUDAERR(cudaFree(gpu_r_[d]));
        free(aux[d]);
        free(dst_[d]);
    }
}
