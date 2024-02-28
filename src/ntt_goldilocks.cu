#include "ntt_goldilocks.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "poseidon_goldilocks.hpp"
#include "ntt_goldilocks.cuh"

// CUDA Threads per Block
#define TPB_V1 64

#define MAX_GPUS 16
gl64_t *gpu_roots[MAX_GPUS];
gl64_t *gpu_a[MAX_GPUS];
gl64_t *gpu_a2[MAX_GPUS];
gl64_t *gpu_forward_twiddle_factors[MAX_GPUS];
gl64_t *gpu_inverse_twiddle_factors[MAX_GPUS];
gl64_t *gpu_r_[MAX_GPUS];
cudaStream_t gpu_stream[MAX_GPUS];
gl64_t *gpu_poseidon_state[MAX_GPUS];

#define GPU_TIMING
// #define GPU_TIMING_2
#ifdef GPU_TIMING
#include "timer.hpp"
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
    /*
    int ind1 = N - i;
    if (ind1 == N)
    {
        ind1 = 0;
    }
    return ind1;
    */
    return (N - i) * (i != 0);
}

__global__ void ntt_iter_loop(uint32_t nBatches, gl64_t *roots, gl64_t *a, gl64_t *a2, gl64_t *r_,
                              uint64_t batchSize, uint64_t ncols, uint64_t rs, uint64_t re, uint64_t rb, uint64_t rm,
                              uint32_t super_s, uint32_t s, uint32_t sInc, uint32_t maxBatchPow, uint32_t domainPow,
                              uint32_t inverse, uint32_t extend, uint32_t size)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x; // tid
    // nBatches is a power of 2, we should not have b >= nbatches
    /*
    if (b >= nBatches)
    {
        return;
    }
    */

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
            mymemcpy((uint64_t *)&a2[offset_dstY], (uint64_t *)&a[offset_src], ncols);
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
    }
}

__global__ void copy_local(uint64_t *dst, uint64_t *src, uint32_t nrows, uint32_t ncols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid
    if (i >= nrows)
        return;

    uint64_t *ldst = dst + i * ncols;
    uint64_t *lsrc = src + i * ncols;
    for (uint32_t j = 0; j < ncols; j++)
    {
        ldst[j] = lsrc[j];
    }
}

__global__ void transpose(uint64_t *dst, uint64_t *src, uint32_t nblocks, uint32_t nrows, uint32_t ncols, uint32_t ncols_last_block)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid
    if (i >= nrows)
        return;

    uint64_t *ldst = dst + i * ((nblocks - 1) * ncols + ncols_last_block);

    for (uint32_t k = 0; k < nblocks - 1; k++)
    {
        for (uint32_t j = 0; j < ncols; j++)
        {
            *ldst = src[k * nrows * ncols + i * ncols + j];
            ldst++;
        }
    }
    // last block
    for (uint32_t j = 0; j < ncols_last_block; j++)
    {
        *ldst = src[(nblocks - 1) * nrows * ncols + i * ncols_last_block + j];
        ldst++;
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
void NTT_Goldilocks::reversePermutation(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, uint32_t numThreads)
{
    uint32_t domainSize = log2(size);
    if (dst != src)
    {
        if (extension <= 1)
        {
#pragma omp parallel for schedule(static) num_threads(numThreads)
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

#pragma omp parallel for schedule(static) num_threads(numThreads)
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
#pragma omp parallel for schedule(static) num_threads(numThreads)
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
            mymemcpy((uint64_t *)&dst[offset_i1], (uint64_t *)&src[offset_r1], ncols);
        }
        else
        {
            u_int64_t ext_ = (size / extension) * ncols_all;

            u_int64_t r = BR(tid, domainSize);
            u_int64_t offset_r1 = r * ncols_all + offset_cols;
            u_int64_t offset_i1 = tid * ncols;
            if (offset_r1 < ext_)
            {
                mymemcpy((uint64_t *)&dst[offset_i1], (uint64_t *)&src[offset_r1], ncols);
            }
            else
            {
                mymemset((uint64_t *)&dst[offset_i1], 0, ncols);
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
                auto tmp = new uint64_t[768];
                mymemcpy((uint64_t *)&tmp[0], (uint64_t *)&src[offset_r], ncols);
                mymemcpy((uint64_t *)&dst[offset_r], (uint64_t *)&src[offset_i], ncols);
                mymemcpy((uint64_t *)&dst[offset_i], (uint64_t *)&tmp[0], ncols);
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
    reversePermutation(tmp, src, size, offset_cols, ncols, ncols_all, nThreads);
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
    reversePermutation(tmp, src, size, offset_cols, ncols, ncols_all, nThreads);
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

        ntt_iter_loop<<<ceil((nBatches) / (1.0 * TPB_V1)), TPB_V1, 0, gpu_stream[gpu_id]>>>(nBatches, gpu_roots[gpu_id], gpu_a[gpu_id], gpu_a2[gpu_id], gpu_r_[gpu_id], batchSize, ncols, rs, re, rb, rm, this->s, s, sInc, maxBatchPow, domainPow, inverse, extend, size);
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

void NTT_Goldilocks::NTT_GPU_iters_onGPU(Goldilocks::Element *dst, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, Goldilocks::Element *aux, bool inverse, bool extend, uint64_t aux_size, bool copyFromGPU, int gpu_id)
{
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
#ifdef GPU_TIMING
    struct timeval start;
    gettimeofday(&start, NULL);
#endif
    reversePermutationGPU<<<ceil(size / (1.0 * TPB_V1)), TPB_V1, 0, gpu_stream[gpu_id]>>>((uint64_t *)tmp, (uint64_t *)gpu_a[gpu_id], size, offset_cols, ncols, ncols_all, extension, log2(size));
#ifdef GPU_TIMING
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
    struct timeval end;
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("NTT_GPU_iters_onGPU: reversePermutationGPU took: %lu ms\n", t / 1000);
#endif
    if (iseven == false)
    {
        tmp = a2;
        a2 = a;
        a = tmp;
    }

    uint64_t actual_size = aux_size;

#ifdef GPU_TIMING
    gettimeofday(&start, NULL);
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

        ntt_iter_loop<<<ceil((nBatches) / (1.0 * TPB_V1)), TPB_V1, 0, gpu_stream[gpu_id]>>>(nBatches, gpu_roots[gpu_id], a, a2, gpu_r_[gpu_id], batchSize, ncols, rs, re, rb, rm, this->s, s, sInc, maxBatchPow, domainPow, inverse, extend, size);
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));

        tmp = a2;
        a2 = a;
        a = tmp;
    }
#ifdef GPU_TIMING
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("NTT_GPU_iters_onGPU: kernels took: %lu ms\n", t / 1000);
#endif

    if (copyFromGPU)
    {
#ifdef GPU_TIMING
        gettimeofday(&start, NULL);
#endif
        CHECKCUDAERR(cudaMemcpyAsync((uint64_t *)dst, (uint64_t *)a, actual_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[gpu_id]));
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
#ifdef GPU_TIMING
        gettimeofday(&end, NULL);
        t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
        printf("NTT_GPU_iters_onGPU: copy data from GPU took: %lu ms\n", t / 1000);
#endif
    }
}

void NTT_Goldilocks::NTT_MultiGPU_iters(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, Goldilocks::Element *aux, bool inverse, bool extend, uint64_t aux_size, uint64_t aux_size_last, int ngpus, int gpu_id, bool copyToGPU, bool copyFromGPU)
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

#ifdef GPU_TIMING_2
    TimerStart(NTT_MultiGPU_iters_reversePermutation);
#endif
    reversePermutation(tmp, src, size, offset_cols, ncols, ncols_all, nThreads / ngpus);
#ifdef GPU_TIMING_2
    TimerStopAndLog(NTT_MultiGPU_iters_reversePermutation);
#endif

    if (iseven == false)
    {
        tmp = a2;
        a2 = a;
        a = tmp;
    }

#ifdef GPU_TIMING_2
    TimerStart(NTT_MultiGPU_iters_CopyToGPU);
#endif
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    uint64_t actual_size = (gpu_id == ngpus - 1) ? aux_size_last : aux_size;
    if (copyToGPU)
    {
        CHECKCUDAERR(cudaMemcpyAsync(gpu_a[gpu_id], (uint64_t *)a, actual_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
        // CHECKCUDAERR(cudaMemcpyAsync(gpu_a2[gpu_id], (uint64_t *)a2, actual_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    }
#ifdef GPU_TIMING_2
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
    TimerStopAndLog(NTT_MultiGPU_iters_CopyToGPU);
    TimerStart(NTT_MultiGPU_iters_Loop);
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

        ntt_iter_loop<<<ceil((nBatches) / (1.0 * TPB_V1)), TPB_V1, 0, gpu_stream[gpu_id]>>>(nBatches, gpu_roots[gpu_id], gpu_a[gpu_id], gpu_a2[gpu_id], gpu_r_[gpu_id], batchSize, ncols, rs, re, rb, rm, this->s, s, sInc, maxBatchPow, domainPow, inverse, extend, size);
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));

        gl64_t *tmpg = gpu_a2[gpu_id];
        gpu_a2[gpu_id] = gpu_a[gpu_id];
        gpu_a[gpu_id] = tmpg;
    }
#ifdef GPU_TIMING_2
    TimerStopAndLog(NTT_MultiGPU_iters_Loop);
#endif

    if (copyFromGPU)
    {
#ifdef GPU_TIMING_2
        TimerStart(NTT_MultiGPU_iters_CopyFromGPU);
#endif
        if (a != dst_)
        {
            a = dst_;
        }
        CHECKCUDAERR(cudaMemcpyAsync((uint64_t *)a, gpu_a[gpu_id], actual_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[gpu_id]));
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
#ifdef GPU_TIMING_2
        TimerStopAndLog(NTT_MultiGPU_iters_CopyFromGPU);
#endif
    }
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
    CHECKCUDAERR(cudaMalloc(&gpu_roots[gpu_id], nRoots * sizeof(uint64_t)));
    if (extend)
    {
        CHECKCUDAERR(cudaMalloc(&gpu_r_[gpu_id], size * sizeof(uint64_t))); // 64M
        CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[gpu_id], (uint64_t *)r_, size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    }
    CHECKCUDAERR(cudaMalloc(&gpu_a[gpu_id], aux_size * sizeof(uint64_t)));  // 42G
    CHECKCUDAERR(cudaMalloc(&gpu_a2[gpu_id], aux_size * sizeof(uint64_t))); // 42G
    CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[gpu_id], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
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
    // #pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        aux[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * size * ncols_per_gpu);
        dst_[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * size * ncols_per_gpu);
    }

#ifdef GPU_TIMING
    TimerStart(NTT_MultiGPU_CopyToGPU_and_Iters);
#endif

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaMalloc(&gpu_roots[d], nRoots * sizeof(uint64_t)));
        if (extend)
        {
            CHECKCUDAERR(cudaMalloc(&gpu_r_[d], size * sizeof(uint64_t)));
            CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[d], (uint64_t *)r_, size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        }
        CHECKCUDAERR(cudaMalloc(&gpu_a[d], aux_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_a2[d], aux_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[d], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
    }

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        NTT_Goldilocks::NTT_MultiGPU_iters(dst_[d], src, size, d * ncols_per_gpu, aux_ncols, ncols, nphase, aux[d], inverse, extend, aux_size, aux_size_last, nDevices, d);
    }
#ifdef GPU_TIMING
    TimerStopAndLog(NTT_MultiGPU_CopyToGPU_and_Iters);
    TimerStart(NTT_MultiGPU_CopyBackFromGPU);
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
    TimerStopAndLog(NTT_MultiGPU_CopyBackFromGPU);
    TimerStart(NTT_MultiGPU_Cleanup);
#endif

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
        CHECKCUDAERR(cudaFree(gpu_roots[d]));
        CHECKCUDAERR(cudaFree(gpu_a[d]));
        CHECKCUDAERR(cudaFree(gpu_a2[d]));
        CHECKCUDAERR(cudaFree(gpu_r_[d]));
        free(aux[d]);
        free(dst_[d]);
    }
#ifdef GPU_TIMING
    TimerStopAndLog(NTT_MultiGPU_Cleanup);
#endif
}

void NTT_Goldilocks::NTT_BatchGPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, u_int64_t ncols_batch, Goldilocks::Element *buffer, u_int64_t nphase, bool inverse, bool extend, bool buildMerkleTree)
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

    printf("*** Batch GPU\n");
    printf("Number of columns: %lu\n", ncols);
    printf("Number of batches: %lu\n", nbatches);
    printf("Cols per batch: %lu\n", ncols_batch);
    printf("Cols last batch: %lu\n", ncols_last_batch);

    Goldilocks::Element *aux;
    Goldilocks::Element *dst_;
    aux = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);
    dst_ = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);

#ifdef GPU_TIMING
    TimerStart(NTT_BatchGPU_PrepareGPU);
#endif

    int gpu_id = 0;
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaStreamCreate(gpu_stream + gpu_id));
    CHECKCUDAERR(cudaMalloc(&gpu_roots[gpu_id], nRoots * sizeof(uint64_t)));
    if (extend)
    {
        CHECKCUDAERR(cudaMalloc(&gpu_r_[gpu_id], size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[gpu_id], (uint64_t *)r_, size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    }
    if (buildMerkleTree)
    {
        assert(ncols_batch % 8 == 0); // ncols per batch should be multiple of 8 which is Poseidon RATE
        CHECKCUDAERR(cudaMalloc(&gpu_poseidon_state[gpu_id], size * SPONGE_WIDTH * sizeof(uint64_t)));
        PoseidonGoldilocks::partial_hash_init_gpu((uint64_t **)gpu_poseidon_state, size, 1);
    }
    CHECKCUDAERR(cudaMalloc(&gpu_a[gpu_id], aux_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_a2[gpu_id], aux_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[gpu_id], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));

#ifdef GPU_TIMING
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
    TimerStopAndLog(NTT_BatchGPU_PrepareGPU);
    TimerStart(NTT_BatchGPU_AllBatches);
#endif

    for (int b = 0; b < nbatches; b++)
    {
        printf("Batch %d\n", b);
#ifdef GPU_TIMING
        TimerStart(NTT_BatchGPU_OneBatch);
#endif
        uint64_t aux_ncols = (b == nbatches - 1 && ncols_last_batch > 0) ? ncols_last_batch : ncols_batch;
        if (buildMerkleTree)
        {
            NTT_Goldilocks::NTT_GPU_iters(dst_, src, size, b * ncols_batch, aux_ncols, ncols, nphase, aux, inverse, extend, aux_size, true, false);
#ifdef GPU_TIMING
            TimerStart(NTT_BatchGPU_PartialHash);
#endif
            PoseidonGoldilocks::partial_hash_gpu((uint64_t *)gpu_a[gpu_id], aux_ncols, size, (uint64_t *)gpu_poseidon_state[gpu_id]);
#ifdef GPU_TIMING
            TimerStopAndLog(NTT_BatchGPU_PartialHash);
#endif
        }
        else
        {
            NTT_Goldilocks::NTT_GPU_iters(dst_, src, size, b * ncols_batch, aux_ncols, ncols, nphase, aux, inverse, extend, aux_size, true, true);
#pragma omp parallel for schedule(static)
            for (u_int64_t ie = 0; ie < size; ++ie)
            {
                u_int64_t offset2 = ie * ncols + b * ncols_batch;
                std::memcpy(&dst[offset2], &(dst_[ie * aux_ncols]), aux_ncols * sizeof(Goldilocks::Element));
            }
        }
#ifdef GPU_TIMING
        TimerStopAndLog(NTT_BatchGPU_OneBatch);
#endif
    }
#ifdef GPU_TIMING
    TimerStopAndLog(NTT_BatchGPU_AllBatches);
#endif

    if (buildMerkleTree)
    {
#ifdef GPU_TIMING
        TimerStart(NTT_BatchGPU_MerkleTree);
#endif
        PoseidonGoldilocks::merkletree_cuda_from_partial(dst, (uint64_t *)gpu_poseidon_state[gpu_id], ncols, size);
#ifdef GPU_TIMING
        TimerStopAndLog(NTT_BatchGPU_MerkleTree);
#endif
    }

#ifdef GPU_TIMING
    TimerStart(NTT_BatchGPU_Cleanup);
#endif
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaStreamDestroy(gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_roots[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a2[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_r_[gpu_id]));
    if (buildMerkleTree)
    {
        CHECKCUDAERR(cudaFree(gpu_poseidon_state[gpu_id]));
    }
    free(aux);
    free(dst_);
#ifdef GPU_TIMING
    TimerStopAndLog(NTT_BatchGPU_Cleanup);
#endif
}

void NTT_Goldilocks::LDE_MerkleTree_BatchGPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, u_int64_t ncols_batch, bool buildMerkleTree)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    this->extension = ext_size / size;

    uint64_t nbatches = ncols / ncols_batch;
    uint64_t ncols_last_batch = ncols - ncols_batch * nbatches;
    if (ncols_last_batch > 0)
    {
        nbatches++;
    }
    uint64_t aux_size = ext_size * ncols_batch;

    printf("\n*** in LDE_BatchGPU() ...\n");
    printf("Number of columns: %lu\n", ncols);
    printf("Number of batches: %lu\n", nbatches);
    printf("Cols per batch: %lu\n", ncols_batch);
    printf("Cols last batch: %lu\n", ncols_last_batch);

    Goldilocks::Element *aux;
    Goldilocks::Element *dst_;
    aux = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);
    dst_ = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);

    int gpu_id = 0;
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaStreamCreate(gpu_stream + gpu_id));
    CHECKCUDAERR(cudaMalloc(&gpu_roots[gpu_id], nRoots * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_r_[gpu_id], size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[gpu_id], (uint64_t *)r_, size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));

    assert(ncols_batch % 8 == 0); // ncols per batch should be multiple of 8 which is Poseidon RATE
    CHECKCUDAERR(cudaMalloc(&gpu_poseidon_state[gpu_id], ext_size * SPONGE_WIDTH * sizeof(uint64_t)));
    PoseidonGoldilocks::partial_hash_init_gpu((uint64_t **)gpu_poseidon_state, ext_size, 1);

    CHECKCUDAERR(cudaMalloc(&gpu_a[gpu_id], aux_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_a2[gpu_id], aux_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[gpu_id], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_BatchGPU_AllBatches);
#endif
    for (int b = 0; b < nbatches; b++)
    {
        printf("Batch %d\n", b);
        uint64_t aux_ncols = (b == nbatches - 1 && ncols_last_batch > 0) ? ncols_last_batch : ncols_batch;

#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_BatchGPU_OneBatch);
#endif
        if (buildMerkleTree)
        {
            // INTT
            NTT_Goldilocks::NTT_GPU_iters(dst_, src, size, b * ncols_batch, aux_ncols, ncols, 3, aux, true, true, size * ncols_batch, true, false);
            // NTT
            NTT_Goldilocks::NTT_GPU_iters_onGPU(dst_, ext_size, 0, aux_ncols, aux_ncols, 3, aux, false, false, ext_size * ncols_batch, false);
            // Merkle
            PoseidonGoldilocks::partial_hash_gpu((uint64_t *)gpu_a[gpu_id], aux_ncols, ext_size, (uint64_t *)gpu_poseidon_state[gpu_id]);
        }
        else
        {
            // INTT
            NTT_Goldilocks::NTT_GPU_iters(dst_, src, size, b * ncols_batch, aux_ncols, ncols, 3, aux, true, true, size * ncols_batch, true, false);
            // NTT
            NTT_Goldilocks::NTT_GPU_iters_onGPU(dst_, ext_size, 0, aux_ncols, aux_ncols, 3, aux, false, false, ext_size * ncols_batch, true);
            // copy back
#pragma omp parallel for schedule(static)
            for (u_int64_t ie = 0; ie < ext_size; ++ie)
            {
                u_int64_t offset2 = ie * ncols + b * ncols_batch;
                std::memcpy(&dst[offset2], &(dst_[ie * aux_ncols]), aux_ncols * sizeof(Goldilocks::Element));
            }
        }
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_BatchGPU_OneBatch);
#endif
    }

#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_BatchGPU_AllBatches);
#endif

    if (buildMerkleTree)
    {
#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_BatchGPU_MerkleTree);
#endif
        PoseidonGoldilocks::merkletree_cuda_from_partial(dst, (uint64_t *)gpu_poseidon_state[gpu_id], ncols, ext_size);
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_BatchGPU_MerkleTree);
#endif
    }

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_BatchGPU_Cleanup);
#endif
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaStreamDestroy(gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_roots[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a2[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_r_[gpu_id]));
    if (buildMerkleTree)
    {
        CHECKCUDAERR(cudaFree(gpu_poseidon_state[gpu_id]));
    }
    free(aux);
    free(dst_);
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_BatchGPU_Cleanup);
#endif
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
        bool toFree = false;
        if (buffer == NULL)
        {
            uint64_t aux_size = ext_size * ncols;
            buffer = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * aux_size);
            toFree = true;
        }
        extendPol(buffer, src, ext_size, size, ncols, NULL, nphase, 1);
        PoseidonGoldilocks::merkletree_avx(dst, buffer, ncols, ext_size);
        if (toFree)
        {
            free(buffer);
        }
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
    CHECKCUDAERR(cudaMalloc(&gpu_r_[gpu_id], size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_a[gpu_id], aux_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_a2[gpu_id], aux_size * sizeof(uint64_t)));

#ifdef GPU_TIMING
    struct timeval start;
    gettimeofday(&start, NULL);
#endif

    CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[gpu_id], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice));
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
    NTT_Goldilocks::NTT_GPU_iters_onGPU(buffer, ext_size, 0, ncols, ncols, nphase, aux, false, false, aux_size, true);

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
    CHECKCUDAERR(cudaFree(gpu_r_[gpu_id]));
    free(aux);
    if (buildMerkleTree)
    {
        free(dst_);
    }
}

void NTT_Goldilocks::LDE_MerkleTree_GPU_v3(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, Goldilocks::Element *buffer, bool buildMerkleTree, u_int64_t nphase)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    printf("*** In LDE_MerkleTree_GPU_v3() ...\n");

    int gpu_id = 0;

    uint64_t aux_size = ext_size * ncols;
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaStreamCreate(&gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaMalloc(&gpu_a[gpu_id], aux_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_forward_twiddle_factors[gpu_id], ext_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_inverse_twiddle_factors[gpu_id], ext_size * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&gpu_r_[gpu_id], ext_size * sizeof(uint64_t)));

    int lg2 = log2(size);
    int lg2ext = log2(ext_size);
    init_twiddle_factors(gpu_forward_twiddle_factors[gpu_id], gpu_inverse_twiddle_factors[gpu_id], lg2);
    init_twiddle_factors(gpu_forward_twiddle_factors[gpu_id], gpu_inverse_twiddle_factors[gpu_id], lg2ext);
    init_r(gpu_r_[gpu_id], lg2);

    CHECKCUDAERR(cudaMemcpyAsync(gpu_a[gpu_id], src, size * ncols * sizeof(gl64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaMemsetAsync(gpu_a[gpu_id] + size * ncols, 0, size * ncols * sizeof(gl64_t), gpu_stream[gpu_id]));
    ntt_cuda(gpu_stream[gpu_id], gpu_a[gpu_id], gpu_r_[gpu_id], gpu_forward_twiddle_factors[gpu_id], gpu_inverse_twiddle_factors[gpu_id], lg2, ncols, true, true);
    ntt_cuda(gpu_stream[gpu_id], gpu_a[gpu_id], gpu_r_[gpu_id], gpu_forward_twiddle_factors[gpu_id], gpu_inverse_twiddle_factors[gpu_id], lg2ext, ncols, false, false);
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));

    if (buildMerkleTree)
    {
        if (buffer != NULL)
        {
            CHECKCUDAERR(cudaMemcpyAsync(buffer, gpu_a[gpu_id], ext_size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToHost, gpu_stream[gpu_id]));
        }
        PoseidonGoldilocks::merkletree_cuda_gpudata(dst, (uint64_t *)gpu_a[gpu_id], ncols, ext_size);
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
    }
    else
    {
        CHECKCUDAERR(cudaMemcpy(dst, gpu_a[gpu_id], ext_size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToHost));
    }

    CHECKCUDAERR(cudaStreamDestroy(gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_a[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_r_[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_forward_twiddle_factors[gpu_id]));
    CHECKCUDAERR(cudaFree(gpu_inverse_twiddle_factors[gpu_id]));
}

void NTT_Goldilocks::LDE_MerkleTree_GPU_v4(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, Goldilocks::Element *buffer, bool buildMerkleTree, u_int64_t nphase)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    printf("*** In LDE_MerkleTree_GPU_v4() ...\n");

    int lg2 = log2(size);
    int lg2ext = log2(ext_size);

    int gpu_id = 0;
    CHECKCUDAERR(cudaSetDevice(gpu_id));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_a[gpu_id], src, size * ncols * sizeof(gl64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaMemsetAsync(gpu_a[gpu_id] + size * ncols, 0, size * ncols * sizeof(gl64_t), gpu_stream[gpu_id]));
    ntt_cuda(gpu_stream[gpu_id], gpu_a[gpu_id], gpu_r_[gpu_id], gpu_forward_twiddle_factors[gpu_id], gpu_inverse_twiddle_factors[gpu_id], lg2, ncols, true, true);
    ntt_cuda(gpu_stream[gpu_id], gpu_a[gpu_id], gpu_r_[gpu_id], gpu_forward_twiddle_factors[gpu_id], gpu_inverse_twiddle_factors[gpu_id], lg2ext, ncols, false, false);
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));

    if (buildMerkleTree)
    {
        if (buffer != NULL)
        {
            CHECKCUDAERR(cudaMemcpyAsync(buffer, gpu_a[gpu_id], ext_size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToHost, gpu_stream[gpu_id]));
        }
        PoseidonGoldilocks::merkletree_cuda_gpudata(dst, (uint64_t *)gpu_a[gpu_id], ncols, ext_size);
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
    }
    else
    {
        CHECKCUDAERR(cudaMemcpy(dst, gpu_a[gpu_id], ext_size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToHost));
    }
}

void NTT_Goldilocks::LDE_MerkleTree_MultiGPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, bool buildMerkleTree)
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
    //    uint64_t aux_ext_size_last = ext_size * ncols_last_gpu;

    printf("*** In Extend_MultiGPU()...\n");
    printf("Number of CPU threads: %d\n", nThreads);
    printf("Number of GPUs: %d\n", nDevices);
    printf("Number columns: %lu\n", ncols);
    printf("Cols per GPU: %lu\n", ncols_per_gpu);
    printf("Cols last GPU: %lu\n", ncols_last_gpu);
    // TODO - we suppose the GPU memory is large enough, so we do not test it

    Goldilocks::Element *aux[MAX_GPUS];
    Goldilocks::Element *dst_[MAX_GPUS];
    // #pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        aux[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * ext_size * ncols_per_gpu);
        dst_[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * ext_size * ncols_per_gpu);
    }

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_PrepareGPUs);
#endif

    // #pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaMalloc(&gpu_roots[d], nRoots * sizeof(uint64_t))); // 64M
        CHECKCUDAERR(cudaMalloc(&gpu_r_[d], ext_size * sizeof(uint64_t)));  // 64M
        CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[d], (uint64_t *)r_, ext_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[d], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMalloc(&gpu_a[d], aux_ext_size * sizeof(uint64_t)));  // 42G
        CHECKCUDAERR(cudaMalloc(&gpu_a2[d], aux_ext_size * sizeof(uint64_t))); // 42G
    }

#ifdef GPU_TIMING
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
    }
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_PrepareGPUs);
#endif

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_INTT);
#endif
    // INTT with extension (inverse and extend set to true)
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        NTT_Goldilocks::NTT_MultiGPU_iters(dst_[d], src, size, d * ncols_per_gpu, aux_ncols, ncols, nphase, aux[d], true, true, aux_size, aux_size_last, nDevices, d, true, false);
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_INTT);
    TimerStart(LDE_MerkleTree_MultiGPU_NTT);
#endif

    // NTT on the extended domain (inverse and extend set to false)
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        // NTT_Goldilocks::NTT_MultiGPU_iters(dst_[d], dst, ext_size, d * ncols_per_gpu, aux_ncols, ncols, nphase, aux[d], false, false, aux_ext_size, aux_ext_size_last, nDevices, d, false, true);
        NTT_Goldilocks::NTT_GPU_iters_onGPU(dst_[d], ext_size, 0, aux_ncols, aux_ncols, nphase, aux[d], false, false, aux_ext_size, !buildMerkleTree, d);
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_NTT);
    TimerStart(LDE_MerkleTree_MultiGPU_PartialCleanup);
#endif

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaFree(gpu_roots[d]));
        CHECKCUDAERR(cudaFree(gpu_a2[d]));
        CHECKCUDAERR(cudaFree(gpu_r_[d]));
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_PartialCleanup);
#endif

    if (buildMerkleTree)
    {
#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_PartialHashInitGPUs);
#endif
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaMalloc(&gpu_poseidon_state[d], ext_size * SPONGE_WIDTH * sizeof(uint64_t)));
        }

        // the function bellow does it for all the GPUs
        PoseidonGoldilocks::partial_hash_init_gpu((uint64_t **)gpu_poseidon_state, ext_size, nDevices);
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_PartialHashInitGPUs);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_PartialHashPeer2Peer);
#endif
        // here we need to go step by step on each GPU
        for (uint32_t d = 0; d < nDevices - 1; d++)
        {
            uint64_t aux_ncols = ncols_per_gpu;
            CHECKCUDAERR(cudaSetDevice(d));
            PoseidonGoldilocks::partial_hash_gpu((uint64_t *)gpu_a[d], aux_ncols, ext_size, (uint64_t *)gpu_poseidon_state[d]);
            CHECKCUDAERR(cudaDeviceSynchronize());
            CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_poseidon_state[d + 1], d + 1, gpu_poseidon_state[d], d, ext_size * SPONGE_WIDTH * sizeof(uint64_t), gpu_stream[d]));
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_PartialHashPeer2Peer);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_PartialHashLastGPU);
#endif
        uint32_t d = nDevices - 1;
        CHECKCUDAERR(cudaSetDevice(d));
        PoseidonGoldilocks::partial_hash_gpu((uint64_t *)gpu_a[d], ncols_last_gpu, ext_size, (uint64_t *)gpu_poseidon_state[d]);
        CHECKCUDAERR(cudaDeviceSynchronize());
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_PartialHashLastGPU);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_MerkleTree);
#endif
        PoseidonGoldilocks::merkletree_cuda_from_partial(dst, (uint64_t *)gpu_poseidon_state[d], ncols_last_gpu, ext_size, d);
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_MerkleTree);
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree);
#endif
    }
    else
    {
#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_MultiGPU_CopyBackAfterNTT);
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
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_CopyBackAfterNTT);
#endif
    }

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_Cleanup);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
        CHECKCUDAERR(cudaFree(gpu_a[d]));
        free(aux[d]);
        free(dst_[d]);
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_Cleanup);
#endif
}

void NTT_Goldilocks::LDE_MerkleTree_MultiGPU_v2(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, bool buildMerkleTree)
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
    //    uint64_t aux_ext_size_last = ext_size * ncols_last_gpu;

    printf("*** In Extend_MultiGPU()...\n");
    printf("Number of CPU threads: %d\n", nThreads);
    printf("Number of GPUs: %d\n", nDevices);
    printf("Number columns: %lu\n", ncols);
    printf("Cols per GPU: %lu\n", ncols_per_gpu);
    printf("Cols last GPU: %lu\n", ncols_last_gpu);
    // TODO - we suppose the GPU memory is large enough, so we do not test it

    Goldilocks::Element *aux[MAX_GPUS];
    Goldilocks::Element *dst_[MAX_GPUS];
    // #pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        aux[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * ext_size * ncols_per_gpu);
        dst_[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * ext_size * ncols_per_gpu);
    }

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_PrepareGPUs);
#endif

    // #pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + nDevices + d));
        CHECKCUDAERR(cudaMalloc(&gpu_roots[d], nRoots * sizeof(uint64_t))); // 64M
        CHECKCUDAERR(cudaMalloc(&gpu_r_[d], ext_size * sizeof(uint64_t)));  // 64M
        CHECKCUDAERR(cudaMemcpyAsync(gpu_r_[d], (uint64_t *)r_, ext_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_roots[d], (uint64_t *)roots, nRoots * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMalloc(&gpu_a[d], aux_ext_size * sizeof(uint64_t)));  // 42G
        CHECKCUDAERR(cudaMalloc(&gpu_a2[d], aux_ext_size * sizeof(uint64_t))); // 42G
    }

#ifdef GPU_TIMING
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
    }
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_PrepareGPUs);
#endif

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_INTT);
#endif
    // INTT with extension (inverse and extend set to true)
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        NTT_Goldilocks::NTT_MultiGPU_iters(dst_[d], src, size, d * ncols_per_gpu, aux_ncols, ncols, nphase, aux[d], true, true, aux_size, aux_size_last, nDevices, d, true, false);
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_INTT);
    TimerStart(LDE_MerkleTree_MultiGPU_NTT);
#endif

    // NTT on the extended domain (inverse and extend set to false)
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        // NTT_Goldilocks::NTT_MultiGPU_iters(dst_[d], dst, ext_size, d * ncols_per_gpu, aux_ncols, ncols, nphase, aux[d], false, false, aux_ext_size, aux_ext_size_last, nDevices, d, false, true);
        NTT_Goldilocks::NTT_GPU_iters_onGPU(dst_[d], ext_size, 0, aux_ncols, aux_ncols, nphase, aux[d], false, false, aux_ext_size, !buildMerkleTree, d);
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_NTT);
    TimerStart(LDE_MerkleTree_MultiGPU_PartialCleanup);
#endif

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaFree(gpu_roots[d]));
        CHECKCUDAERR(cudaFree(gpu_r_[d]));
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_PartialCleanup);
#endif

    if (buildMerkleTree)
    {
        uint64_t nrows_per_gpu = ext_size / nDevices;
        uint64_t nrows_last_gpu = nrows_per_gpu;
        if (ext_size % nDevices != 0)
        {
            nrows_last_gpu = ext_size - (nDevices - 1) * nrows_per_gpu;
        }
        printf("Rows per GPU: %lu\n", nrows_per_gpu);
        printf("Rows last GPU: %lu\n", nrows_last_gpu);
        uint64_t block_elem = nrows_per_gpu * ncols_per_gpu;
        uint64_t block_size = block_elem * sizeof(uint64_t);

#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_Peer2PeerCopy);
#endif
        // di is destination, dj is source
        for (uint64_t di = 0; di < nDevices; di++)
        {
            for (uint64_t dj = 0; dj < nDevices - 1; dj++)
            {
                CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_a2[di] + dj * block_elem, di, gpu_a[dj] + di * block_elem, dj, block_size, gpu_stream[di]));
                /*
                if (di != dj)
                {
                    // P2P
                    CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_a2[di] + dj * block_elem, di, gpu_a[dj] + di * block_elem, dj, block_size, gpu_stream[di]));
                }
                else
                {
                    // local
                    copy_local<<<ceil(nrows_per_gpu / (1.0 * TPB_V1)), TPB_V1, 0, gpu_stream[di]>>>((uint64_t*)gpu_a2[di] + di * block_elem, (uint64_t*)gpu_a[di] + di * block_elem, nrows_per_gpu, ncols_per_gpu);
                }
                */
            }
        }
        // last block may have different size
        uint64_t block_elem_last = nrows_per_gpu * ncols_last_gpu;
        uint64_t block_size_last = block_elem_last * sizeof(uint64_t);
        for (uint64_t di = 0; di < nDevices; di++)
        {
            uint64_t dj = nDevices - 1;
            CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_a2[di] + dj * block_elem, di, gpu_a[dj] + di * block_elem_last, dj, block_size_last, gpu_stream[di]));
        }
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_Peer2PeerCopy);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_Transpose);
#endif
        // re-arrange (transpose)
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            transpose<<<ceil(nrows_per_gpu / (1.0 * TPB_V1)), TPB_V1, 0, gpu_stream[d]>>>((uint64_t *)gpu_a[d], (uint64_t *)gpu_a2[d], nDevices, nrows_per_gpu, ncols_per_gpu, ncols_last_gpu);
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }

#pragma omp parallel for num_threads(nDevices)
        for (uint64_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            // CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
            CHECKCUDAERR(cudaMemcpyAsync(buffer + d * (nrows_per_gpu * ncols), gpu_a[d], nrows_per_gpu * ncols * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[nDevices + d]));
        }
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[nDevices + d]));
        }
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_Transpose);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_Kernel);
#endif
        // Merkle tree building
        PoseidonGoldilocks::merkletree_cuda_multi_gpu_full(dst, (uint64_t **)gpu_a, (uint64_t **)gpu_a2, gpu_stream, ncols, ext_size, nrows_per_gpu, nDevices);
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_Kernel);
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree);
#endif
    }
    else
    {
#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_MultiGPU_CopyBackAfterNTT);
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
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_CopyBackAfterNTT);
#endif
    }

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_Cleanup);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[nDevices + d]));
        CHECKCUDAERR(cudaFree(gpu_a[d]));
        CHECKCUDAERR(cudaFree(gpu_a2[d]));
        free(aux[d]);
        free(dst_[d]);
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_Cleanup);
#endif
}

void NTT_Goldilocks::LDE_MerkleTree_MultiGPU_v3(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, bool buildMerkleTree)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    uint64_t ncols_per_gpu = (ncols % nDevices) ? ncols / nDevices + 1 : ncols / nDevices;
    uint64_t ncols_last_gpu = ncols - ncols_per_gpu * (nDevices - 1);
    uint64_t aux_ext_size = ext_size * ncols_per_gpu;

    printf("*** In LDE_MerkleTree_MultiGPU_v3() ...\n");
    printf("Number of CPU threads: %d\n", nThreads);
    printf("Number of GPUs: %d\n", nDevices);
    printf("Number columns: %lu\n", ncols);
    printf("Cols per GPU: %lu\n", ncols_per_gpu);
    printf("Cols last GPU: %lu\n", ncols_last_gpu);
    // TODO - we suppose the GPU memory is large enough, so we do not test it

    int lg2 = log2(size);
    int lg2ext = log2(ext_size);

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_PrepareGPUs);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + nDevices + d));
        CHECKCUDAERR(cudaMalloc(&gpu_a[d], aux_ext_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_r_[d], ext_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_forward_twiddle_factors[d], ext_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_inverse_twiddle_factors[d], ext_size * sizeof(uint64_t)));
        init_twiddle_factors(gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2);
        init_twiddle_factors(gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2ext);
        init_r(gpu_r_[d], lg2);
    }

#ifdef GPU_TIMING
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
    }
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_PrepareGPUs);
#endif

    Goldilocks::Element *aux[MAX_GPUS];
#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_SplitColsOnCPU);
#endif
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        if (buffer != NULL)
        {
            aux[d] = buffer + d * ext_size * ncols_per_gpu;
        }
        else
        {
            aux[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * ext_size * aux_ncols);
        }
        assert(aux[d] != NULL);
#pragma omp parallel for schedule(static)
        for (u_int64_t ie = 0; ie < size; ++ie)
        {
            u_int64_t offset2 = ie * ncols + d * ncols_per_gpu;
            std::memcpy(&(aux[d][ie * aux_ncols]), &src[offset2], aux_ncols * sizeof(Goldilocks::Element));
        }
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_SplitColsOnCPU);
#endif

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_v3_EXT);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        CHECKCUDAERR(cudaMemcpyAsync(gpu_a[d], aux[d], size * aux_ncols * sizeof(gl64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemsetAsync(gpu_a[d] + size * aux_ncols, 0, size * aux_ncols * sizeof(gl64_t), gpu_stream[d]));
        ntt_cuda(gpu_stream[d], gpu_a[d], gpu_r_[d], gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2, aux_ncols, true, true);
        ntt_cuda(gpu_stream[d], gpu_a[d], gpu_r_[d], gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2ext, aux_ncols, false, false);
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_v3_EXT);
    TimerStart(LDE_MerkleTree_MultiGPU_PartialCleanup);
#endif

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaFree(gpu_forward_twiddle_factors[d]));
        CHECKCUDAERR(cudaFree(gpu_inverse_twiddle_factors[d]));
        CHECKCUDAERR(cudaFree(gpu_r_[d]));
        CHECKCUDAERR(cudaMalloc(&gpu_a2[d], aux_ext_size * sizeof(uint64_t)));
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_PartialCleanup);
#endif

    if (buildMerkleTree)
    {
        uint64_t nrows_per_gpu = ext_size / nDevices;
        uint64_t nrows_last_gpu = nrows_per_gpu;
        if (ext_size % nDevices != 0)
        {
            nrows_last_gpu = ext_size - (nDevices - 1) * nrows_per_gpu;
        }
        printf("Rows per GPU: %lu\n", nrows_per_gpu);
        printf("Rows last GPU: %lu\n", nrows_last_gpu);
        uint64_t block_elem = nrows_per_gpu * ncols_per_gpu;
        uint64_t block_size = block_elem * sizeof(uint64_t);

#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_Peer2PeerCopy);
#endif
        // di is destination, dj is source
        for (uint64_t di = 0; di < nDevices; di++)
        {
            for (uint64_t dj = 0; dj < nDevices - 1; dj++)
            {
                CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_a2[di] + dj * block_elem, di, gpu_a[dj] + di * block_elem, dj, block_size, gpu_stream[di]));
            }
        }
        // last block may have different size
        uint64_t block_elem_last = nrows_per_gpu * ncols_last_gpu;
        uint64_t block_size_last = block_elem_last * sizeof(uint64_t);
        for (uint64_t di = 0; di < nDevices; di++)
        {
            uint64_t dj = nDevices - 1;
            CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_a2[di] + dj * block_elem, di, gpu_a[dj] + di * block_elem_last, dj, block_size_last, gpu_stream[di]));
        }
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_Peer2PeerCopy);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_Transpose);
#endif
        // re-arrange (transpose)
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            transpose<<<ceil(nrows_per_gpu / (1.0 * TPB_V1)), TPB_V1, 0, gpu_stream[d]>>>((uint64_t *)gpu_a[d], (uint64_t *)gpu_a2[d], nDevices, nrows_per_gpu, ncols_per_gpu, ncols_last_gpu);
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }

        if (buffer != NULL)
        {
#pragma omp parallel for num_threads(nDevices)
            for (uint64_t d = 0; d < nDevices; d++)
            {
                CHECKCUDAERR(cudaSetDevice(d));
                // CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
                CHECKCUDAERR(cudaMemcpyAsync(buffer + d * (nrows_per_gpu * ncols), gpu_a[d], nrows_per_gpu * ncols * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[nDevices + d]));
            }
        }

#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaFree(gpu_a2[d]));
            CHECKCUDAERR(cudaMalloc(&gpu_a2[d], 4 * nrows_per_gpu * sizeof(uint64_t)));
        }

// sync memcpy
        if (buffer != NULL)
        {
#pragma omp parallel for num_threads(nDevices)
            for (uint32_t d = 0; d < nDevices; d++)
            {
                CHECKCUDAERR(cudaSetDevice(d));
                CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[nDevices + d]));
            }
        }

#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_Transpose);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_Kernel);
#endif
        // Merkle tree building
        PoseidonGoldilocks::merkletree_cuda_multi_gpu_full(dst, (uint64_t **)gpu_a, (uint64_t **)gpu_a2, gpu_stream, ncols, ext_size, nrows_per_gpu, nDevices);
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_Kernel);
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree);
#endif
    }
    else
    {
#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_MultiGPU_CopyBackAfterNTT);
#endif
        for (uint32_t d = 0; d < nDevices; d++)
        {
            uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
#pragma omp parallel for schedule(static)
            for (u_int64_t ie = 0; ie < ext_size; ++ie)
            {
                u_int64_t offset2 = ie * ncols + d * ncols_per_gpu;
                // std::memcpy(&dst[offset2], &(aux[d][ie * aux_ncols]), aux_ncols * sizeof(Goldilocks::Element));
                CHECKCUDAERR(cudaMemcpyAsync(&dst[offset2], &gpu_a[d][ie * aux_ncols], aux_ncols * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[d]));
            }
        }
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_CopyBackAfterNTT);
#endif
    }

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_Cleanup);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[nDevices + d]));
        CHECKCUDAERR(cudaFree(gpu_a[d]));
        CHECKCUDAERR(cudaFree(gpu_a2[d]));
        if (buffer == NULL)
        {
            free(aux[d]);
        }
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_Cleanup);
#endif
}

void NTT_Goldilocks::LDE_MerkleTree_MultiGPU_v3_viaCPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, bool buildMerkleTree)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    uint64_t ncols_per_gpu = (ncols % nDevices) ? ncols / nDevices + 1 : ncols / nDevices;
    uint64_t ncols_last_gpu = ncols - ncols_per_gpu * (nDevices - 1);
    uint64_t aux_ext_size = ext_size * ncols_per_gpu;

    printf("*** In LDE_MerkleTree_MultiGPU_v3_viaCPU() ...\n");
    printf("Number of CPU threads: %d\n", nThreads);
    printf("Number of GPUs: %d\n", nDevices);
    printf("Number columns: %lu\n", ncols);
    printf("Cols per GPU: %lu\n", ncols_per_gpu);
    printf("Cols last GPU: %lu\n", ncols_last_gpu);
    // TODO - we suppose the GPU memory is large enough, so we do not test it

    int lg2 = log2(size);
    int lg2ext = log2(ext_size);

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_PrepareGPUs);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + nDevices + d));
        CHECKCUDAERR(cudaMalloc(&gpu_a[d], aux_ext_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_r_[d], ext_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_forward_twiddle_factors[d], ext_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_inverse_twiddle_factors[d], ext_size * sizeof(uint64_t)));
        init_twiddle_factors(gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2);
        init_twiddle_factors(gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2ext);
        init_r(gpu_r_[d], lg2);
    }

#ifdef GPU_TIMING
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
    }
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_PrepareGPUs);
#endif

    Goldilocks::Element *aux[MAX_GPUS];
#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_SplitColsOnCPU);
#endif
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        if (buffer != NULL)
        {
            aux[d] = buffer + d * ext_size * ncols_per_gpu;
        }
        else
        {
            aux[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * ext_size * aux_ncols);
        }
        assert(aux[d] != NULL);
#pragma omp parallel for schedule(static)
        for (u_int64_t ie = 0; ie < size; ++ie)
        {
            u_int64_t offset2 = ie * ncols + d * ncols_per_gpu;
            std::memcpy(&(aux[d][ie * aux_ncols]), &src[offset2], aux_ncols * sizeof(Goldilocks::Element));
        }
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_SplitColsOnCPU);
#endif

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_v3_EXT);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        CHECKCUDAERR(cudaMemcpyAsync(gpu_a[d], aux[d], size * aux_ncols * sizeof(gl64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemsetAsync(gpu_a[d] + size * aux_ncols, 0, size * aux_ncols * sizeof(gl64_t), gpu_stream[d]));
        ntt_cuda(gpu_stream[d], gpu_a[d], gpu_r_[d], gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2, aux_ncols, true, true);
        ntt_cuda(gpu_stream[d], gpu_a[d], gpu_r_[d], gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2ext, aux_ncols, false, false);
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_v3_EXT);
    TimerStart(LDE_MerkleTree_MultiGPU_PartialCleanup);
#endif

    uint64_t* buffer2 = (uint64_t*)malloc(ext_size * ncols * sizeof(uint64_t));
    assert(NULL != buffer2);

#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaFree(gpu_forward_twiddle_factors[d]));
        CHECKCUDAERR(cudaFree(gpu_inverse_twiddle_factors[d]));
        CHECKCUDAERR(cudaFree(gpu_r_[d]));
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_PartialCleanup);
#endif

    if (buildMerkleTree)
    {
        uint64_t nrows_per_gpu = ext_size / nDevices;
        uint64_t nrows_last_gpu = nrows_per_gpu;
        if (ext_size % nDevices != 0)
        {
            nrows_last_gpu = ext_size - (nDevices - 1) * nrows_per_gpu;
        }
        printf("Rows per GPU: %lu\n", nrows_per_gpu);
        printf("Rows last GPU: %lu\n", nrows_last_gpu);

#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_D2H);
#endif

        // Transpose is done on CPU. First we copy data to CPU.
        assert(buffer != NULL);
#pragma omp parallel for num_threads(nDevices)
        for (uint64_t d = 0; d < nDevices; d++)
        {
            uint64_t ncols_act = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
            CHECKCUDAERR(cudaMemcpyAsync(buffer2 + d * ext_size * ncols_per_gpu, gpu_a[d], ext_size * ncols_act * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[d]));
        }
#pragma omp parallel for num_threads(nDevices)
        for (uint64_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }

#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_D2H);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_TransposeOnCPU);
#endif

#pragma omp parallel for
        for (uint64_t row = 0; row < ext_size; row++)
        {
            uint64_t* dst = (uint64_t*)buffer + row * ncols;
            for (uint64_t d = 0; d < nDevices - 1; d++)
            {
                uint64_t* src = buffer2 + d * ext_size * ncols_per_gpu + row * ncols_per_gpu;
                memcpy(dst + d * ncols_per_gpu, src, ncols_per_gpu * sizeof(uint64_t));
            }
            // last block
            uint64_t d = nDevices - 1;
            uint64_t* src = buffer2 + d * ext_size * ncols_per_gpu + row * ncols_last_gpu;
            memcpy(dst + d * ncols_per_gpu, src, ncols_last_gpu * sizeof(uint64_t));
        }
        free(buffer2);

#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_TransposeOnCPU);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_H2D);
#endif

#pragma omp parallel for num_threads(nDevices)
        for (uint64_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaMemcpyAsync(gpu_a[d], buffer + d * nrows_per_gpu * ncols, nrows_per_gpu * ncols * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        }

#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_H2D);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_Kernel);
#endif

#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaMalloc(&gpu_a2[d], 4 * nrows_per_gpu * sizeof(uint64_t)));
        }

        // Merkle tree building
        PoseidonGoldilocks::merkletree_cuda_multi_gpu_full(dst, (uint64_t **)gpu_a, (uint64_t **)gpu_a2, gpu_stream, ncols, ext_size, nrows_per_gpu, nDevices);
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_Kernel);
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree);
#endif
    }
    else
    {
#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_MultiGPU_CopyBackAfterNTT);
#endif
        for (uint32_t d = 0; d < nDevices; d++)
        {
            uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
#pragma omp parallel for schedule(static)
            for (u_int64_t ie = 0; ie < ext_size; ++ie)
            {
                u_int64_t offset2 = ie * ncols + d * ncols_per_gpu;
                // std::memcpy(&dst[offset2], &(aux[d][ie * aux_ncols]), aux_ncols * sizeof(Goldilocks::Element));
                CHECKCUDAERR(cudaMemcpyAsync(&dst[offset2], &gpu_a[d][ie * aux_ncols], aux_ncols * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[d]));
            }
        }
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_CopyBackAfterNTT);
#endif
    }

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_Cleanup);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[nDevices + d]));
        CHECKCUDAERR(cudaFree(gpu_a[d]));
        CHECKCUDAERR(cudaFree(gpu_a2[d]));
        if (buffer == NULL)
        {
            free(aux[d]);
        }
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_Cleanup);
#endif
}

void NTT_Goldilocks::LDE_MerkleTree_MultiGPU_Init(u_int64_t size, u_int64_t ext_size, u_int64_t ncols)
{
    int lg2 = log2(size);
    int lg2ext = log2(ext_size);

    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));

    uint64_t ncols_per_gpu = (ncols % nDevices) ? ncols / nDevices + 1 : ncols / nDevices;
    uint64_t aux_ext_size = ext_size * ncols_per_gpu;

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_PrepareGPUs);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + nDevices + d));
        CHECKCUDAERR(cudaMalloc(&gpu_a[d], aux_ext_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_a2[d], aux_ext_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_r_[d], ext_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_forward_twiddle_factors[d], ext_size * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&gpu_inverse_twiddle_factors[d], ext_size * sizeof(uint64_t)));
        init_twiddle_factors(gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2);
        init_twiddle_factors(gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2ext);
        init_r(gpu_r_[d], lg2);
    }

#ifdef GPU_TIMING
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
    }
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_PrepareGPUs);
#endif
}

void NTT_Goldilocks::LDE_MerkleTree_MultiGPU_Free()
{
    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_Cleanup);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[nDevices + d]));
        CHECKCUDAERR(cudaFree(gpu_a[d]));
        CHECKCUDAERR(cudaFree(gpu_a2[d]));
        CHECKCUDAERR(cudaFree(gpu_forward_twiddle_factors[d]));
        CHECKCUDAERR(cudaFree(gpu_inverse_twiddle_factors[d]));
        CHECKCUDAERR(cudaFree(gpu_r_[d]));
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_Cleanup);
#endif
}

void NTT_Goldilocks::LDE_MerkleTree_MultiGPU_v4(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ext_size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, bool buildMerkleTree)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    uint64_t ncols_per_gpu = (ncols % nDevices) ? ncols / nDevices + 1 : ncols / nDevices;
    uint64_t ncols_last_gpu = ncols - ncols_per_gpu * (nDevices - 1);
    uint64_t aux_ext_size = ext_size * ncols_per_gpu;

    printf("*** In LDE_MerkleTree_MultiGPU_v4() ...\n");
    printf("Number of GPUs: %d\n", nDevices);
    printf("Number columns: %lu\n", ncols);
    printf("Cols per GPU: %lu\n", ncols_per_gpu);
    printf("Cols last GPU: %lu\n", ncols_last_gpu);
    // TODO - we suppose the GPU memory is large enough, so we do not test it

    int lg2 = log2(size);
    int lg2ext = log2(ext_size);

    bool realloc_a2 = false;

    Goldilocks::Element *aux[MAX_GPUS];
#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_SplitColsOnCPU);
#endif
    for (uint32_t d = 0; d < nDevices; d++)
    {
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        if (buffer != NULL)
        {
            aux[d] = buffer + d * ext_size * ncols_per_gpu;
        }
        else
        {
            aux[d] = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * ext_size * aux_ncols);
        }
        assert(aux[d] != NULL);
#pragma omp parallel for schedule(static)
        for (u_int64_t ie = 0; ie < size; ++ie)
        {
            u_int64_t offset2 = ie * ncols + d * ncols_per_gpu;
            std::memcpy(&(aux[d][ie * aux_ncols]), &src[offset2], aux_ncols * sizeof(Goldilocks::Element));
        }
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_SplitColsOnCPU);
#endif

#ifdef GPU_TIMING
    TimerStart(LDE_MerkleTree_MultiGPU_v3_EXT);
#endif
#pragma omp parallel for num_threads(nDevices)
    for (uint32_t d = 0; d < nDevices; d++)
    {
        CHECKCUDAERR(cudaSetDevice(d));
        uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
        CHECKCUDAERR(cudaMemcpyAsync(gpu_a[d], aux[d], size * aux_ncols * sizeof(gl64_t), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemsetAsync(gpu_a[d] + size * aux_ncols, 0, size * aux_ncols * sizeof(gl64_t), gpu_stream[d]));
        ntt_cuda(gpu_stream[d], gpu_a[d], gpu_r_[d], gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2, aux_ncols, true, true);
        ntt_cuda(gpu_stream[d], gpu_a[d], gpu_r_[d], gpu_forward_twiddle_factors[d], gpu_inverse_twiddle_factors[d], lg2ext, aux_ncols, false, false);
        CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
    }
#ifdef GPU_TIMING
    TimerStopAndLog(LDE_MerkleTree_MultiGPU_v3_EXT);
#endif

    if (buildMerkleTree)
    {
        uint64_t nrows_per_gpu = ext_size / nDevices;
        uint64_t nrows_last_gpu = nrows_per_gpu;
        if (ext_size % nDevices != 0)
        {
            nrows_last_gpu = ext_size - (nDevices - 1) * nrows_per_gpu;
        }
        printf("Rows per GPU: %lu\n", nrows_per_gpu);
        printf("Rows last GPU: %lu\n", nrows_last_gpu);
        uint64_t block_elem = nrows_per_gpu * ncols_per_gpu;
        uint64_t block_size = block_elem * sizeof(uint64_t);

#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_Peer2PeerCopy);
#endif
        // di is destination, dj is source
        for (uint64_t di = 0; di < nDevices; di++)
        {
            for (uint64_t dj = 0; dj < nDevices - 1; dj++)
            {
                CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_a2[di] + dj * block_elem, di, gpu_a[dj] + di * block_elem, dj, block_size, gpu_stream[di]));
            }
        }
        // last block may have different size
        uint64_t block_elem_last = nrows_per_gpu * ncols_last_gpu;
        uint64_t block_size_last = block_elem_last * sizeof(uint64_t);
        for (uint64_t di = 0; di < nDevices; di++)
        {
            uint64_t dj = nDevices - 1;
            CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_a2[di] + dj * block_elem, di, gpu_a[dj] + di * block_elem_last, dj, block_size_last, gpu_stream[di]));
        }
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_Peer2PeerCopy);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_Transpose);
#endif
        // re-arrange (transpose)
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            transpose<<<ceil(nrows_per_gpu / (1.0 * TPB_V1)), TPB_V1, 0, gpu_stream[d]>>>((uint64_t *)gpu_a[d], (uint64_t *)gpu_a2[d], nDevices, nrows_per_gpu, ncols_per_gpu, ncols_last_gpu);
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }

        if (buffer != NULL)
        {
#pragma omp parallel for num_threads(nDevices)
            for (uint64_t d = 0; d < nDevices; d++)
            {
                CHECKCUDAERR(cudaSetDevice(d));
                // CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
                CHECKCUDAERR(cudaMemcpyAsync(buffer + d * (nrows_per_gpu * ncols), gpu_a[d], nrows_per_gpu * ncols * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[nDevices + d]));
            }
        }

        if (aux_ext_size >= 1409286144)
        {
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaFree(gpu_a2[d]));
            CHECKCUDAERR(cudaMalloc(&gpu_a2[d], 4 * nrows_per_gpu * sizeof(uint64_t)));
        }
        realloc_a2 = true;
        }

// sync memcpy
        if (buffer != NULL)
        {
#pragma omp parallel for num_threads(nDevices)
            for (uint32_t d = 0; d < nDevices; d++)
            {
                CHECKCUDAERR(cudaSetDevice(d));
                CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[nDevices + d]));
            }
        }

#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_Transpose);
        TimerStart(LDE_MerkleTree_MultiGPU_MerkleTree_Kernel);
#endif
        // Merkle tree building
        PoseidonGoldilocks::merkletree_cuda_multi_gpu_full(dst, (uint64_t **)gpu_a, (uint64_t **)gpu_a2, gpu_stream, ncols, ext_size, nrows_per_gpu, nDevices);
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree_Kernel);
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_MerkleTree);
#endif
    }
    else
    {
#ifdef GPU_TIMING
        TimerStart(LDE_MerkleTree_MultiGPU_CopyBackAfterNTT);
#endif
        for (uint32_t d = 0; d < nDevices; d++)
        {
            uint64_t aux_ncols = (d == nDevices - 1) ? ncols_last_gpu : ncols_per_gpu;
#pragma omp parallel for schedule(static)
            for (u_int64_t ie = 0; ie < ext_size; ++ie)
            {
                u_int64_t offset2 = ie * ncols + d * ncols_per_gpu;
                // std::memcpy(&dst[offset2], &(aux[d][ie * aux_ncols]), aux_ncols * sizeof(Goldilocks::Element));
                CHECKCUDAERR(cudaMemcpyAsync(&dst[offset2], &gpu_a[d][ie * aux_ncols], aux_ncols * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[d]));
            }
        }
#pragma omp parallel for num_threads(nDevices)
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
#ifdef GPU_TIMING
        TimerStopAndLog(LDE_MerkleTree_MultiGPU_CopyBackAfterNTT);
#endif
    }

    if (buffer == NULL)
    {
        for (uint32_t d = 0; d < nDevices; d++)
        {
            free(aux[d]);
        }
    }
    if (realloc_a2)
    {
        for (uint32_t d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaMalloc(&gpu_a2[d], aux_ext_size * sizeof(uint64_t)));
        }
    }
}