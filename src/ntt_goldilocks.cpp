#include "ntt_goldilocks.hpp"
#include <sys/time.h>
//#ifdef __USE_CUDA__
//#include <src/lib.h>
//#include <cuda/ntt/ntt.h>
//#endif // __USE_CUDA__

static inline u_int64_t BR(u_int64_t x, u_int64_t domainPow)
{
  x = (x >> 16) | (x << 16);
  x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
  x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
  x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
  return (((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)) >> (32 - domainPow);
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
      printf("\nafter reversePermutation:\n");
      printf("[");
      for (uint j = 0; j < size * ncols && j<8; j++)
      {
          printf("%lu, ", Goldilocks::toU64(tmp[j]));
      }
      printf("]\n");

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
  computeR(N);

  INTT(output, input, N, ncols, tmp, nphase, nblock, true);
  printf("intt output:\n");
  printf("[");
  for (uint32_t i = 0; i <N*2 && i<8; i++) {
    printf("%lu, ", Goldilocks::toU64(output[i]));
  }
  printf("]\n");
  ntt_extension.NTT(output, output, N_Extended, ncols, tmp, nphase, nblock);

  if (buffer == NULL)
  {
    free(tmp);
  }
}

//#ifdef __USE_CUDA__
//void NTT_Goldilocks::NTT_cuda(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, Goldilocks::Element *buffer, bool transpose)
//{
//  Goldilocks::Element *data = NULL;
//  if (transpose && ncols > 1) {
//    if (buffer != NULL)
//    {
//      data = buffer;
//    }
//    else
//    {
//      data = (Goldilocks::Element *)malloc(size * ncols * sizeof(Goldilocks::Element));
//    }
//
//#pragma omp parallel for schedule(static)
//    for (u_int64_t j = 0; j < size; j++) {
//      for (u_int64_t i = 0; i < ncols; i++) {
//        data[i*size+j] = src[j*ncols+i];
//      }
//    }
//  } else {
//    data = src;
//  }
//
//  uint64_t batch_size = ncols / TOTAL_GPU;
//  uint64_t last_batch_size = batch_size + ncols % TOTAL_GPU;
//#pragma omp parallel for schedule(static)
//  for (u_int32_t i = 0; i < TOTAL_GPU; i++) {
//    u_int64_t par_ncols = i == (TOTAL_GPU-1)? last_batch_size: batch_size;
//    if (par_ncols > 0) {
//        compute_batched_ntt(i, (fr_t *)(uint64_t *)(data+size*batch_size*i), log2(size), par_ncols, Ntt_Types::NN, Ntt_Types::Direction::forward, Ntt_Types::Type::standard);
//    }
//  }
//
//  Goldilocks::Element *dst_ = NULL;
//  if (transpose && ncols > 1) {
//    if (dst != NULL && dst != src)
//    {
//      dst_ = dst;
//    }
//    else
//    {
//      dst_ = src;
//    }
//
//#pragma omp parallel for schedule(static)
//    for (u_int64_t j = 0; j < size; j++) {
//      for (u_int64_t i = 0; i < ncols; i++) {
//        dst_[j*ncols+i] = data[i*size+j];
//      }
//    }
//  } else if (dst != NULL && dst != src) {
//    Goldilocks::parcpy(dst, src, size * ncols, nThreads);
//  }
//}
//
//void NTT_Goldilocks::INTT_cuda(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, Goldilocks::Element *buffer, bool transpose, bool extend)
//{
//  struct timeval start, end;
//  Goldilocks::Element *data = NULL;
//  if (transpose && ncols > 1) {
//    if (buffer != NULL)
//    {
//      data = buffer;
//    }
//    else
//    {
//      data = (Goldilocks::Element *)malloc(size * ncols * sizeof(Goldilocks::Element));
//    }
//
//    // transform from row-major to column-major order
//#pragma omp parallel for schedule(static)
//    for (u_int64_t j = 0; j < size; j++) {
//      for (u_int64_t i = 0; i < ncols; i++) {
//        data[i*size+j] = src[j*ncols+i];
//      }
//    }
//  } else {
//    data = src;
//  }
//
//  gettimeofday(&start, NULL);
//  uint64_t batch_size = ncols / TOTAL_GPU;
//  uint64_t last_batch_size = batch_size + ncols % TOTAL_GPU;
//#pragma omp parallel for schedule(static)
//  for (u_int32_t i = 0; i < TOTAL_GPU; i++) {
//    u_int64_t par_ncols = i == (TOTAL_GPU-1)? last_batch_size: batch_size;
//    if (par_ncols > 0) {
//      compute_batched_ntt(i, (fr_t *)(uint64_t *)(data+size*batch_size*i), log2(size), par_ncols, Ntt_Types::NN, Ntt_Types::Direction::inverse, Ntt_Types::Type::standard);
//    }
//  }
//  gettimeofday(&end, NULL);
//  long seconds = end.tv_sec - start.tv_sec;
//  long microseconds = end.tv_usec - start.tv_usec;
//  long elapsed = seconds*1000 + microseconds/1000;
//  std::cout << "intt elapsed: " << elapsed << " ms\n";
//
//  if (extend) {
//    gettimeofday(&start, NULL);
//#pragma omp parallel for schedule(static)
//    for (u_int64_t j = 0; j < size; j++) {
//      for (u_int64_t i = 0; i < ncols; i++) {
//        data[i*size+j] = data[i*size+j] * r[j];
//      }
//    }
//    gettimeofday(&end, NULL);
//    long seconds = end.tv_sec - start.tv_sec;
//    long microseconds = end.tv_usec - start.tv_usec;
//    long elapsed = seconds*1000 + microseconds/1000;
//    std::cout << "intt extend elapsed: " << elapsed << " ms\n";
//  }
//
//  Goldilocks::Element *dst_ = NULL;
//  if (transpose && ncols > 1) {
//    if (dst != NULL && dst != src)
//    {
//      dst_ = dst;
//    }
//    else
//    {
//      dst_ = src;
//    }
//
//
//#pragma omp parallel for schedule(static)
//    for (u_int64_t j = 0; j < size; j++) {
//      for (u_int64_t i = 0; i < ncols; i++) {
//        dst_[j*ncols+i] = data[i*size+j];
//      }
//    }
//
//    if (buffer == NULL) {
//      free(data);
//    }
//  } else if (dst != NULL && dst != src) {
//    Goldilocks::parcpy(dst, src, size * ncols, nThreads);
//  }
//}
//
//void NTT_Goldilocks::extendPol_cuda(Goldilocks::Element *dst, Goldilocks::Element *src, uint64_t N_Extended, uint64_t N, uint64_t ncols, Goldilocks::Element *buffer, bool transpose)
//{
//  printf("extendPol_cuda, src=%p, dst=%p, log_N_Extended=%d, log_N=%d, ncols=%ld; buffer=%p, transpose=%d, nThreads=%d\n", dst, src, log2(N_Extended), log2(N), ncols, buffer, transpose, nThreads);
//  struct timeval start, end;
//  if (dst == NULL) {
//      dst = src;
//  }
//  Goldilocks::Element *data = NULL;
//  if (transpose && ncols > 1) {
//    gettimeofday(&start, NULL);
//    if (buffer != NULL)
//    {
//      data = buffer;
//    }
//    else
//    {
//      data = (Goldilocks::Element *)malloc(N_Extended * ncols * sizeof(Goldilocks::Element));
//    }
//
//    // transform from row-major to column-major order
//#pragma omp parallel for schedule(static)
//    for (u_int64_t j = 0; j < N; j++) {
//      for (u_int64_t i = 0; i < ncols; i++) {
//        data[i*N+j] = src[j*ncols+i];
//      }
//    }
//    gettimeofday(&end, NULL);
//    long seconds = end.tv_sec - start.tv_sec;
//    long microseconds = end.tv_usec - start.tv_usec;
//    long elapsed = seconds*1000 + microseconds/1000;
//    std::cout << "transpose1 elapsed: " << elapsed << " ms\n";
//  } else {
//    data = src;
//  }
//
//  INTT_cuda(data, data, N, ncols, NULL, false, true);
//
////      printf("\nINTT outputs:\n");
////      printf("[");
////      for (uint j = 0; j < N_Extended * ncols; j++)
////      {
////          printf("%lu, ", Goldilocks::toU64(data[j]));
////      }
////      printf("]\n");
//
//  gettimeofday(&start, NULL);
////  // can not be done in parallel, but the second half can be done in parallel
////  for (u_int64_t j = ncols-1; j > 0; j--) {
////    std::memcpy(&data[j * N_Extended], &data[j * N], N * sizeof(Goldilocks::Element));
////  }
//
//  for (u_int64_t j = ncols; j > 1; j=(j+1)/2) {
//#pragma omp parallel for
//    for (u_int64_t i = j-1; i > (j-1)/2; i--) {
//      std::memcpy(&data[i * N_Extended], &data[i * N], N * sizeof(Goldilocks::Element));
//    }
//  }
//  gettimeofday(&end, NULL);
//  long seconds = end.tv_sec - start.tv_sec;
//  long microseconds = end.tv_usec - start.tv_usec;
//  long elapsed = seconds*1000 + microseconds/1000;
//  std::cout << "memcpy1 elapsed: " << elapsed << " ms\n";
//
//  gettimeofday(&start, NULL);
//#pragma omp parallel for schedule(static)
//  for (u_int64_t i = 0; i < ncols; i++) {
//    std::memcpy(&data[i * N_Extended + N], &data[(ncols-1) * N_Extended + N], N * sizeof(Goldilocks::Element));
//  }
//  gettimeofday(&end, NULL);
//  seconds = end.tv_sec - start.tv_sec;
//  microseconds = end.tv_usec - start.tv_usec;
//  elapsed = seconds*1000 + microseconds/1000;
//  std::cout << "memcpy2 elapsed: " << elapsed << " ms\n";
//
//  //    printf("\nINTT2 outputs:\n");
//  //    printf("[");
//  //    for (uint j = 0; j < N_Extended * ncols; j++)
//  //    {
//  //        printf("%lu, ", Goldilocks::toU64(data[j]));
//  //    }
//  //    printf("]\n");
//
//  gettimeofday(&start, NULL);
//  NTT_cuda(data, data, N_Extended, ncols, NULL, false);
//  gettimeofday(&end, NULL);
//  seconds = end.tv_sec - start.tv_sec;
//  microseconds = end.tv_usec - start.tv_usec;
//  elapsed = seconds*1000 + microseconds/1000;
//  std::cout << "ntt elapsed: " << elapsed << " ms\n";
//
//  if (transpose && ncols > 1) {
//    struct timeval start, end;
//    gettimeofday(&start, NULL);
//
//#pragma omp parallel for schedule(static)
//    for (u_int64_t j = 0; j < N_Extended; j++) {
//      for (u_int64_t i = 0; i < ncols; i++) {
//        dst[j*ncols+i] = data[i*N_Extended+j];
//        //dst[i*N_Extended+j] = data[j*ncols+i];
//      }
//    }
//    gettimeofday(&end, NULL);
//    long seconds = end.tv_sec - start.tv_sec;
//    long microseconds = end.tv_usec - start.tv_usec;
//    long elapsed = seconds*1000 + microseconds/1000;
//    std::cout << "transpose2 elapsed: " << elapsed << " ms\n";
//
//    if (buffer == NULL) {
//      free(data);
//    }
//  } else if (dst != src) {
//    Goldilocks::parcpy(dst, src, N_Extended * ncols, nThreads);
//  }
//}
//
//// 8 gpu, log_domain_size = 30
//static bool twiddle_factor_flags[240];
//
//void NTT_Goldilocks::init_twiddle_factors_cuda(u_int32_t device_id, u_int32_t lg_n)
//{
//  if (!twiddle_factor_flags[device_id*30 + lg_n]) {
//    twiddle_factor_flags[device_id*30 + lg_n] = true;
//    init_twiddle_factors(device_id, lg_n);
//  }
//}
//#endif // __USE_CUDA__
