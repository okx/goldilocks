#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "ntt_goldilocks.hpp"
#include <cuda_runtime.h>
#include <sys/time.h>

__device__ __constant__ uint64_t omegas[33] = {
    1,
    18446744069414584320ULL,
    281474976710656ULL,
    16777216ULL,
    4096ULL,
    64ULL,
    8ULL,
    2198989700608ULL,
    4404853092538523347ULL,
    6434636298004421797ULL,
    4255134452441852017ULL,
    9113133275150391358ULL,
    4355325209153869931ULL,
    4308460244895131701ULL,
    7126024226993609386ULL,
    1873558160482552414ULL,
    8167150655112846419ULL,
    5718075921287398682ULL,
    3411401055030829696ULL,
    8982441859486529725ULL,
    1971462654193939361ULL,
    6553637399136210105ULL,
    8124823329697072476ULL,
    5936499541590631774ULL,
    2709866199236980323ULL,
    8877499657461974390ULL,
    3757607247483852735ULL,
    4969973714567017225ULL,
    2147253751702802259ULL,
    2530564950562219707ULL,
    1905180297017055339ULL,
    3524815499551269279ULL,
    7277203076849721926ULL,
};

__device__ __constant__ uint64_t omegas_inv[33] = {
    0x1,
    0xffffffff00000000,
    0xfffeffff00000001,
    0xfffffeff00000101,
    0xffefffff00100001,
    0xfbffffff04000001,
    0xdfffffff20000001,
    0x3fffbfffc0,
    0x7f4949dce07bf05d,
    0x4bd6bb172e15d48c,
    0x38bc97652b54c741,
    0x553a9b711648c890,
    0x55da9bb68958caa,
    0xa0a62f8f0bb8e2b6,
    0x276fd7ae450aee4b,
    0x7b687b64f5de658f,
    0x7de5776cbda187e9,
    0xd2199b156a6f3b06,
    0xd01c8acd8ea0e8c0,
    0x4f38b2439950a4cf,
    0x5987c395dd5dfdcf,
    0x46cf3d56125452b1,
    0x909c4b1a44a69ccb,
    0xc188678a32a54199,
    0xf3650f9ddfcaffa8,
    0xe8ef0e3e40a92655,
    0x7c8abec072bb46a6,
    0xe0bfc17d5c5a7a04,
    0x4c6b8a5a0b79f23a,
    0x6b4d20533ce584fe,
    0xe5cceae468a70ec2,
    0x8958579f296dac7a,
    0x16d265893b5b7e85,
};

__device__ __constant__ uint64_t domain_size_inverse[33] = {
    0x0000000000000001,  // 1^{-1}
    0x7fffffff80000001,  // 2^{-1}
    0xbfffffff40000001,  // (1 << 2)^{-1}
    0xdfffffff20000001,  // (1 << 3)^{-1}
    0xefffffff10000001,
    0xf7ffffff08000001,
    0xfbffffff04000001,
    0xfdffffff02000001,
    0xfeffffff01000001,
    0xff7fffff00800001,
    0xffbfffff00400001,
    0xffdfffff00200001,
    0xffefffff00100001,
    0xfff7ffff00080001,
    0xfffbffff00040001,
    0xfffdffff00020001,
    0xfffeffff00010001,
    0xffff7fff00008001,
    0xffffbfff00004001,
    0xffffdfff00002001,
    0xffffefff00001001,
    0xfffff7ff00000801,
    0xfffffbff00000401,
    0xfffffdff00000201,
    0xfffffeff00000101,
    0xffffff7f00000081,
    0xffffffbf00000041,
    0xffffffdf00000021,
    0xffffffef00000011,
    0xfffffff700000009,
    0xfffffffb00000005,
    0xfffffffd00000003,
    0xfffffffe00000002,  // (1 << 32)^{-1}
};

// CUDA Threads Per Block
#define TPB 16
#define SHIFT 7
#define MAX_LOG_ITEMS 8


__device__ gl64_t FORWARD_TWIDDLE_FACTORS[1<<24];
__device__ gl64_t INVERSE_TWIDDLE_FACTORS[1<<24];
__device__ gl64_t r[1<<24];


__global__ void br_ntt_group(gl64_t *data, uint32_t i, uint32_t domain_size, uint32_t ncols, gl64_t *twiddles) {
  uint32_t j = blockIdx.x;
  uint32_t col = threadIdx.x;
  if (j < domain_size/2 && col < ncols) {
    uint32_t half_group_size = 1<<i;
    uint32_t group = j>>i; // j/(group_size/2);
    uint32_t offset = j&(half_group_size-1); //j%(half_group_size);
    uint32_t index1 = (group<<i+1) + offset;
    uint32_t index2 = index1 + half_group_size;
    gl64_t factor = twiddles[offset * (domain_size>>i+1)];
    gl64_t odd_sub = data[index2*ncols+col] * factor;
    data[index2*ncols+col] = data[index1*ncols+col] - odd_sub;
    data[index1*ncols+col] = data[index1*ncols+col] + odd_sub;
  }
}

__global__ void intt_scale(gl64_t *data, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols, gl64_t *r) {
  uint32_t j = blockIdx.x; // domain_size
  uint32_t col = threadIdx.x; //cols
  uint32_t index = j*ncols + col;
  gl64_t factor = gl64_t(domain_size_inverse[log_domain_size]);
  if (r != NULL) {
    factor *= r[j];
  }
  if (index < domain_size*ncols) {
    data[index] = data[index] * factor;
  }
}

//__global__ void br_ntt_group_single(gl64_t *data, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols) {
//    for (uint32_t i = 0; i < log_domain_size; i++) {
//        uint32_t half_group_size = 1 << i;
//        for (uint32_t j = 0; j < domain_size / 2; j++) {
//            uint32_t group = j >> i; // j/(group_size/2);
//            uint32_t offset = j & (half_group_size - 1); //j%(half_group_size);
//            uint32_t index1 = (group << i + 1) + offset;
//            uint32_t index2 = index1 + half_group_size;
//            for (uint32_t col = 0; col < ncols; col++) {
//                gl64_t odd_sub = data[index2 * ncols + col] * FORWARD_TWIDDLE_FACTORS[offset * domain_size >> i + 1];
//                data[index2 * ncols + col] = data[index1 * ncols + col] - odd_sub;
//                data[index1 * ncols + col] = data[index1 * ncols + col] + odd_sub;
//            }
//        }
//    }
//}
//
//__global__ void br_ntt_group_single2(gl64_t *data, uint32_t i, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols) {
//  uint32_t j = blockIdx.x;
//  uint32_t col = threadIdx.x;
//  printf("bidx: %d, tidx: %d\n", j, col);
//  if (j < domain_size/2 && col < ncols) {
//    uint32_t half_group_size = 1<<i;
//    uint32_t group = j>>i; // j/(group_size/2);
//    uint32_t offset = j&(half_group_size-1); //j%(half_group_size);
//    uint32_t index1 = (group<<i+1) + offset;
//    uint32_t index2 = index1 + half_group_size;
//    gl64_t odd_sub = data[index2*ncols+col] * FORWARD_TWIDDLE_FACTORS[offset * domain_size>>i+1];
//    data[index2*ncols+col] = data[index1*ncols+col] - odd_sub;
//    data[index1*ncols+col] = data[index1*ncols+col] + odd_sub;
//  }
//}

__global__ void reverse_permutation(gl64_t *data, uint32_t log_domain_size, uint32_t ncols) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t ibr = __brev(idx) >> (32-log_domain_size);
  if (ibr > idx) {
    gl64_t tmp;
    for (uint32_t i = 0; i<ncols;i++) {
      tmp = data[idx*ncols+i];
      data[idx*ncols+i] = data[ibr*ncols+i];
      data[ibr*ncols+i] = tmp;
    }
  }
}

//__global__ void init_twiddle_factors(uint32_t log_domain_size) {
//  FORWARD_TWIDDLE_FACTORS[0] = gl64_t::one();
//  INVERSE_TWIDDLE_FACTORS[0] = gl64_t::one();
//  r[0] = gl64_t::one();
//  for (uint32_t i = 1; i<(1<<log_domain_size); i++) {
//    FORWARD_TWIDDLE_FACTORS[i] = FORWARD_TWIDDLE_FACTORS[i - 1] * gl64_t(omegas[log_domain_size]);
//    INVERSE_TWIDDLE_FACTORS[i] = INVERSE_TWIDDLE_FACTORS[i - 1] * gl64_t(omegas_inv[log_domain_size]);
//    r[i] = r[i-1] * gl64_t(SHIFT);
//  }
//}
//
//__global__ void load_tf(gl64_t* dst, uint32_t log_domain_size) {
//  for (uint32_t i = 0; i<(1<<log_domain_size); i++) {
//    dst[i] = FORWARD_TWIDDLE_FACTORS[i];
//  }
//}

__global__ void init_twiddle_factors_small_size(gl64_t* twiddles, uint32_t log_domain_size, bool inverse) {
  gl64_t omega;
  if (inverse) {
    omega = gl64_t(omegas_inv[log_domain_size]);
  } else {
    omega = gl64_t(omegas[log_domain_size]);
  }
  twiddles[0] = gl64_t::one();
  for (uint32_t i = 1; i<(1<<log_domain_size); i++) {
    twiddles[i] = twiddles[i-1] * omega;
  }
}

__global__ void init_twiddle_factors_first_step(gl64_t* twiddles, uint32_t log_domain_size, bool inverse) {
  gl64_t omega;
  if (inverse) {
    omega = gl64_t(omegas_inv[log_domain_size]);
  } else {
    omega = gl64_t(omegas[log_domain_size]);
  }
  twiddles[0] = gl64_t::one();
  for (uint32_t i = 1; i<=1<<12; i++) {
    twiddles[i] = twiddles[i-1] * omega;
  }
}

__global__ void init_twiddle_factors_second_step(gl64_t* twiddles, uint32_t log_domain_size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = 1; i<1<<log_domain_size-12; i++) {
    twiddles[i*4096 + idx] = twiddles[(i-1)*4096 + idx] * twiddles[4096];
  }
}



__global__ void init_twiddle_factors(gl64_t* twiddles, uint32_t log_domain_size, bool inverse) {
  if (log_domain_size <= 12) {
    init_twiddle_factors_small_size<<<1, 1>>>(twiddles, log_domain_size, inverse);
  } else {
    init_twiddle_factors_first_step<<<1, 1>>>(twiddles, log_domain_size, inverse);
    init_twiddle_factors_second_step<<<1<<12, 1>>>(twiddles, log_domain_size);
  }
}

__global__ void init_r_small_size(gl64_t *r, uint32_t log_domain_size) {
  r[0] = gl64_t::one();
  for (uint32_t i = 1; i<(1<<log_domain_size); i++) {
    r[i] = r[i-1] * gl64_t(SHIFT);
  }
}

__global__ void init_r_first_step(gl64_t *r) {
  r[0] = gl64_t::one();
  // init first 4097 elements and then init others in parallel
  for (uint32_t i = 1; i<=1<<12; i++) {
    r[i] = r[i-1] * gl64_t(SHIFT);
  }
}

__global__ void init_r_second_step(gl64_t *r, uint32_t log_domain_size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = 1; i<1<<log_domain_size-12; i++) {
    r[i*4096 + idx] = r[(i-1)*4096 + idx] * r[4096];
  }

}

void init_r(gl64_t *r, uint32_t log_domain_size) {
  if (log_domain_size <= 12) {
    init_r_small_size<<<1,1>>>(r, log_domain_size);
    CHECKCUDAERR(cudaGetLastError());
  } else {
    init_r_first_step<<<1, 1>>>(r);
    CHECKCUDAERR(cudaGetLastError());
    init_r_second_step<<<1<<12, 1>>>(r, log_domain_size);
    CHECKCUDAERR(cudaGetLastError());
  }
}

void ntt_cuda(cudaStream_t stream, gl64_t *data, uint32_t log_domain_size, uint32_t ncols, bool inverse, gl64_t *twiddles, gl64_t *r) {

  uint32_t domain_size = 1<<log_domain_size;

  dim3 blockDim;
  dim3 gridDim;
  if (domain_size > TPB) {
    blockDim = dim3(TPB);
    gridDim = dim3(domain_size/TPB);
  } else {
    blockDim = dim3(domain_size);
    gridDim = dim3(1);
  }

  struct timeval start, end;
  gettimeofday(&start, NULL);
  reverse_permutation<<<gridDim, blockDim, 0, stream>>>(data, log_domain_size, ncols);
  CHECKCUDAERR(cudaGetLastError());
  cudaStreamSynchronize(stream);
  gettimeofday(&end, NULL);
  long seconds = end.tv_sec - start.tv_sec;
  long microseconds = end.tv_usec - start.tv_usec;
  long elapsed = seconds*1000 + microseconds/1000;
  printf("reverse_permutation elapsed: %ld ms\n", elapsed);

#ifdef  __PRINT_LOG__
  uint64_t *log = (uint64_t *)malloc(MAX_LOG_ITEMS * sizeof(uint64_t));
  CHECKCUDAERR(cudaMemcpy(log, data, MAX_LOG_ITEMS * sizeof(gl64_t), cudaMemcpyDeviceToHost));
  printf("\nrp outputs:\n");
  printf("[");
  for (uint j = 0; j < domain_size * ncols && j < MAX_LOG_ITEMS; j++)
  {
    printf("%lu, ", log[j]);
  }
  printf("]\n");
  free(log);
#endif

  gettimeofday(&start, NULL);
  //dim3 blockIdx = dim3(domain_size/2);
  for (uint32_t i = 0; i < log_domain_size; i++) {
    br_ntt_group<<<domain_size/2, ncols, 0, stream>>>(data, i, domain_size, ncols, twiddles);
  }
  cudaStreamSynchronize(stream);
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds*1000 + microseconds/1000;
  printf("br_ntt_group elapsed: %ld ms\n", elapsed);
  if (inverse) {
    gettimeofday(&start, NULL);
    intt_scale<<<domain_size, ncols, 0, stream>>>(data, domain_size, log_domain_size, ncols, r);
    cudaStreamSynchronize(stream);
    gettimeofday(&end, NULL);
    seconds = end.tv_sec - start.tv_sec;
    microseconds = end.tv_usec - start.tv_usec;
    elapsed = seconds*1000 + microseconds/1000;
    printf("intt_scale elapsed: %ld ms\n", elapsed);
  }

}

void init_input(Goldilocks::Element *a, uint32_t domain_size, uint32_t ncols) {
  for (uint64_t i = 0; i < 2; i++)
  {
    for (uint64_t j = 0; j < ncols; j++)
    {
      a[i * ncols + j] = Goldilocks::fromU64(1+j);
    }
  }

  for (uint64_t i = 2; i < domain_size; i++)
  {
    for (uint64_t j = 0; j < ncols; j++)
    {
      a[i * ncols + j] = a[ncols * (i - 1) + j] + a[ncols * (i - 2) + j];
    }
  }
}

//void init_twiddle_factors_cuda(u_int32_t device_id, u_int32_t log_domain_size) {
//  CHECKCUDAERR(cudaSetDevice(device_id));
//  init_twiddle_factors<<<1,1>>>(log_domain_size);
//  CHECKCUDAERR(cudaGetLastError());
//
//}

void NTT_cuda(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, gl64_t *twiddles) {
  uint32_t domain_size = 1<<size;
  gl64_t *d_data;
  CHECKCUDAERR(cudaMalloc((void**)&d_data, domain_size * ncols * sizeof(gl64_t)));
  CHECKCUDAERR(cudaMemcpy(d_data, src, domain_size * ncols * sizeof(gl64_t), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  ntt_cuda(stream, d_data, size, ncols, false, twiddles, NULL);

  cudaStreamDestroy(stream);

  CHECKCUDAERR(cudaMemcpy(dst, d_data, domain_size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToHost));
  cudaFree(d_data);
}

void INTT_cuda(Goldilocks::Element *dst, Goldilocks::Element *src, u_int64_t size, u_int64_t ncols, gl64_t *twiddles, gl64_t *r) {
  uint32_t domain_size = 1<<size;
  gl64_t *d_data;
  CHECKCUDAERR(cudaMalloc((void**)&d_data, domain_size * ncols * sizeof(gl64_t)));
  CHECKCUDAERR(cudaMemcpy(d_data, src, domain_size * ncols * sizeof(gl64_t), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  ntt_cuda(stream, d_data, size, ncols, true, twiddles, r);

  cudaStreamDestroy(stream);


  CHECKCUDAERR(cudaMemcpy(dst, d_data, domain_size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToHost));
  cudaFree(d_data);
}

void extendPol_cuda(Goldilocks::Element *dst, Goldilocks::Element *src, uint64_t log_N_Extended, uint64_t log_N, uint64_t ncols, gl64_t *twiddles_ext, gl64_t *twiddles, gl64_t *r) {
  struct timeval start, end;
  gettimeofday(&start, NULL);

  uint32_t domain_size = 1<<log_N;
  uint32_t domain_size_ext = 1<<log_N_Extended;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  gl64_t *d_data;
  CHECKCUDAERR(cudaMallocAsync((void**)&d_data, domain_size_ext * ncols * sizeof(gl64_t), stream));
  gettimeofday(&end, NULL);
  long seconds = end.tv_sec - start.tv_sec;
  long microseconds = end.tv_usec - start.tv_usec;
  long elapsed = seconds*1000 + microseconds/1000;
  printf("malloc elapsed: %ld ms\n", elapsed);

  gettimeofday(&start, NULL);

  CHECKCUDAERR(cudaMemcpyAsync(d_data, src, domain_size * ncols * sizeof(gl64_t), cudaMemcpyHostToDevice, stream));
  cudaStreamSynchronize(stream);
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds*1000 + microseconds/1000;
  printf("memcpy elapsed: %ld ms\n", elapsed);

  CHECKCUDAERR(cudaMemsetAsync(d_data + domain_size * ncols, 0, domain_size * ncols * sizeof(gl64_t), stream));

  ntt_cuda(stream, d_data, log_N, ncols, true, twiddles, r);

#ifdef  __PRINT_LOG__
  uint64_t *log = (uint64_t *)malloc(MAX_LOG_ITEMS * sizeof(uint64_t));
  CHECKCUDAERR(cudaMemcpy(log, d_data, MAX_LOG_ITEMS * sizeof(gl64_t), cudaMemcpyDeviceToHost));
  printf("\nintt outputs:\n");
  printf("[");
  for (uint j = 0; j < domain_size * ncols && j < MAX_LOG_ITEMS; j++)
  {
    printf("%lu, ", log[j]);
  }
  printf("]\n");
  free(log);
#endif

  ntt_cuda(stream, d_data, log_N_Extended, ncols, false, twiddles_ext, NULL);

  cudaStreamSynchronize(stream);

  {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    uint32_t batch = 64;
    for (uint32_t i = 0; i < batch; i++) {
      CHECKCUDAERR(cudaMemcpyAsync(dst + domain_size_ext * ncols/batch, d_data + domain_size_ext * ncols/batch, domain_size_ext * ncols/batch * sizeof(gl64_t), cudaMemcpyDeviceToHost, stream));
    }
    cudaFree(d_data);

    gettimeofday(&end, NULL);
    seconds = end.tv_sec - start.tv_sec;
    microseconds = end.tv_usec - start.tv_usec;
    elapsed = seconds*1000 + microseconds/1000;
    printf("cudaMemcpy elapsed: %ld ms\n", elapsed);
  }

  cudaStreamDestroy(stream);

  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds*1000 + microseconds/1000;
  printf("extendPol_cuda total elapsed: %ld ms\n", elapsed);

}

int main() {
  CHECKCUDAERR(cudaSetDevice(0)); 
  uint32_t log_domain_size = 23;
  uint32_t domain_size = 1<<log_domain_size;
  uint32_t ncols = 84;
  //uint64_t *a;
  //cudaHostAlloc((void**)&a, 2*domain_size * ncols * sizeof(uint64_t), cudaHostAllocDefault);
  uint64_t *a = (uint64_t *)malloc(2*domain_size * ncols * sizeof(uint64_t));
  uint64_t *b = (uint64_t *)malloc(2*domain_size * ncols * sizeof(uint64_t));
  init_input((Goldilocks::Element *)a, domain_size, ncols);

#ifdef __PRINT_LOG__
  printf("\ninputs:\n");
  printf("[");
  for (uint j = 0; j < domain_size * ncols && j < MAX_LOG_ITEMS; j++)
  {
    printf("%lu, ", a[j]);
  }
  printf("]\n");
#endif

  gl64_t *forward_twiddle_factors;
  CHECKCUDAERR(cudaMalloc((void**)&forward_twiddle_factors, 2*domain_size * sizeof(gl64_t)));
  init_twiddle_factors<<<1,1>>>(forward_twiddle_factors, log_domain_size+1, false);
  CHECKCUDAERR(cudaGetLastError());

  gl64_t *inverse_twiddle_factors;
  CHECKCUDAERR(cudaMalloc((void**)&inverse_twiddle_factors, domain_size * sizeof(gl64_t)));
  init_twiddle_factors<<<1,1>>>(inverse_twiddle_factors, log_domain_size, true);
  CHECKCUDAERR(cudaGetLastError());

  gl64_t *r;
  CHECKCUDAERR(cudaMalloc((void**)&r, domain_size * sizeof(gl64_t)));
  init_r(r, log_domain_size);
  CHECKCUDAERR(cudaGetLastError());


  extendPol_cuda((Goldilocks::Element *)b, (Goldilocks::Element *)a, log_domain_size+1, log_domain_size, ncols, forward_twiddle_factors, inverse_twiddle_factors, r);

#ifdef __PRINT_LOG__
  printf("\noutputs:\n");
  printf("[");
  for (uint j = 0; j < 2*domain_size * ncols && j<8; j++)
  {
    printf("%lu, ", b[j]);
  }
  printf("]\n");
#endif

  cudaFree(forward_twiddle_factors);
  cudaFree(inverse_twiddle_factors);
  cudaFree(r);
  free(a);
  //cudaFreeHost(a);
  free(b);

}
