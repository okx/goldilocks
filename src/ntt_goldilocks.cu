#include "ntt_goldilocks.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"

// CUDA Threads per Block
#define TPB 64
#define MAX_GPUS 16

gl64_t *gpu_roots[16];
gl64_t *gpu_a[16];
gl64_t *gpu_a2[16];
gl64_t *gpu_powTwoInv[16];
gl64_t *gpu_r_[16];
cudaStream_t gpu_stream[16];

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

void NTT_Goldilocks::NTT_GPU_iters(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t offset_cols, u_int64_t ncols, u_int64_t ncols_all, u_int64_t nphase, Goldilocks::Element *aux, bool inverse, bool extend, uint64_t aux_size, uint64_t aux_size_last, int ngpus, int gpu_id)
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
    CHECKCUDAERR(cudaMemcpyAsync(gpu_a[gpu_id], (uint64_t *)a + gpu_id * aux_size, actual_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_a2[gpu_id], (uint64_t *)a2 + gpu_id + aux_size, actual_size * sizeof(uint64_t), cudaMemcpyHostToDevice, gpu_stream[gpu_id]));

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

    CHECKCUDAERR(cudaMemcpyAsync((uint64_t *)a + gpu_id * aux_size, gpu_a[gpu_id], actual_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, gpu_stream[gpu_id]));
    CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[gpu_id]));
}

void NTT_Goldilocks::NTT_GPU(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, bool inverse, bool extend)
{
    if (ncols == 0 || size == 0)
    {
        return;
    }

    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    uint64_t ncols_per_gpu = ncols / nDevices + 1;
    uint64_t ncols_last_gpu = ncols - ncols_per_gpu * (nDevices - 1);
    uint64_t aux_size = size * ncols_per_gpu;
    uint64_t aux_size_last = size * ncols_last_gpu;

    printf("Number of GPUs: %d\n", nDevices);
    printf("Cols per GPU: %lu\n", ncols_per_gpu);
    printf("Cols last GPU: %lu\n", ncols_last_gpu);
    // TODO - we suppose the GPU memory is large enough

    Goldilocks::Element *aux = NULL;
    if (buffer == NULL)
    {
        aux = (Goldilocks::Element *)malloc(sizeof(Goldilocks::Element) * size * ncols);
    }
    else
    {
        aux = buffer;
    }

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
        NTT_Goldilocks::NTT_GPU_iters(dst, src, size, d * ncols_per_gpu, aux_ncols, ncols, nphase, aux, inverse, extend, aux_size, aux_size_last, nDevices, d);
    }

    if (buffer == NULL)
    {
        free(aux);
    }

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

void NTT_Goldilocks::extendPol(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t N_Extended, uint64_t N, uint64_t ncols, Goldilocks::Element *buffer, u_int64_t nphase, u_int64_t nblock)
{
    NTT_Goldilocks ntt_extension(N_Extended, nThreads, N_Extended / N);
    ntt_extension.setUseGPU(this->use_gpu);

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

    if (use_gpu)
    {
        INTT(output, input, N, ncols, tmp, nphase, true);
        ntt_extension.NTT(output, output, N_Extended, ncols, tmp, nphase);
    }
    else
    {
        INTT(output, input, N, ncols, tmp, nphase, nblock, true);
        ntt_extension.NTT(output, output, N_Extended, ncols, tmp, nphase, nblock);
    }

    if (buffer == NULL)
    {
        free(tmp);
    }
}