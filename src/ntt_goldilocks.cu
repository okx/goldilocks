#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "ntt_goldilocks.hpp"
#include <cuda_runtime.h>

__device__ __constant__ uint64_t omegas[33] = {
    1
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

// CUDA Threads Per Block
#define TPB 1

__device__ gl64_t* FORWARD_TWIDDLE_FACTORS[33];
__device__ gl64_t* INVERSE_TWIDDLE_FACTORS[33];

//__global__ void br_ntt_group(gl64_t *data, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols) {
//  for (uint32_t i = 0; i< log_domain_size; i++) {
//    uint32_t half_group_size = 1<<i;
//    for (uint32_t j = 0; j< domain_size/2; j++) {
//      uint32_t group = j>>i; // j/(group_size/2);
//      uint32_t offset = j&(half_group_size-1); //j%(half_group_size);
//      uint32_t index1 = (group<<i+1) + offset;
//      uint32_t index2 = index1 + half_group_size;
//      for (uint32_t col = 0; col < ncols; col++) {
//        gl64_t odd_sub = multiply(dst[index2*ncols+col], FORWARD_TWIDDLE_FACTORS[log_domain_size][offset * domain_size>>i+1]);
//        dst[index2*ncols+col] = dst[index1*ncols+col] - odd_sub;
//        dst[index1*ncols+col] = dst[index1*ncols+col] + odd_sub;
//      }
//    }
//  }
//}

__global__ void br_ntt_group(gl64_t *data, uint32_t half_log_group_size, uint32_t domain_size, uint32_t log_domain_size, uint32_t ncols) {
  uint32_t i = half_log_group_size;        // assert(half_log_group_size < log_domain_size);
  uint32_t j = blockIdx.x;   // assert(blockIdx.y < domain_size / 2);
  uint32_t col = threadIdx.x;               // assert(blockIdx.x < ncols);
  uint32_t half_group_size = 1<<i;
  uint32_t group = j>>i; // j/(group_size/2);
  uint32_t offset = j&(half_group_size-1); //j%(half_group_size);
  uint32_t index1 = (group<<i+1) + offset;
  uint32_t index2 = index1 + half_group_size;
  gl64_t odd_sub = data[index2*ncols+col] *FORWARD_TWIDDLE_FACTORS[log_domain_size][offset * domain_size>>i+1];
  data[index2*ncols+col] = data[index1*ncols+col] - odd_sub;
  data[index1*ncols+col] = data[index1*ncols+col] + odd_sub;
}

__global__ void reverse_permutation(gl64_t *data, uint32_t log_domain_size, uint32_t ncols) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t ibr = __brev(idx) >> (32-log_domain_size);
  printf("idx: %d, ibr: %d\n", idx, ibr);
  if (ibr > idx) {
    for (uint32_t i = 0; i<ncols;i++) {
      gl64_t tmp = data[idx*ncols+i];
      data[idx*ncols+i] = data[ibr*ncols+i];
      data[ibr*ncols+i] = tmp;
    }
  }
}

__global__ void init_twiddle_factors(uint32_t log_domain_size) {
  printf("into init_twiddle_factors\n");
  tif (FORWARD_TWIDDLE_FACTORS[log_domain_size] == NULL) {
    cudaMalloc((void**)&FORWARD_TWIDDLE_FACTORS[log_domain_size], (1<<log_domain_size)*sizeof(gl64_t));
    FORWARD_TWIDDLE_FACTORS[log_domain_size][0] = gl64_t::one();
    for (uint32_t i = 1; i<(1<<log_domain_size); i++) {
      FORWARD_TWIDDLE_FACTORS[log_domain_size][i] = FORWARD_TWIDDLE_FACTORS[log_domain_size][i-1] * gl64_t(omegas[log_domain_size]);
    }
  }
  if (INVERSE_TWIDDLE_FACTORS[log_domain_size] == NULL) {
    cudaMalloc((void**)&INVERSE_TWIDDLE_FACTORS[log_domain_size], (1<<log_domain_size)*sizeof(gl64_t));
    INVERSE_TWIDDLE_FACTORS[log_domain_size][0] = gl64_t::one();
    for (uint32_t i = 1; i<(1<<log_domain_size); i++) {
      INVERSE_TWIDDLE_FACTORS[log_domain_size][i] = INVERSE_TWIDDLE_FACTORS[log_domain_size][i-1] * gl64_t(omegas_inv[log_domain_size]);
    }
  }

  printf("\ntf:\n");
  printf("[");
  for (uint j = 0; j < (1<<log_domain_size); j++)
  {
    uint64_t tmp = uint64_t(FORWARD_TWIDDLE_FACTORS[log_domain_size][j]);
    printf("%lu, ", tmp);
  }
  printf("]\n");
}

void NTT_cuda(gl64_t *dst, gl64_t *src, uint32_t log_domain_size, uint32_t ncols) {
  uint32_t domain_size = 1<<log_domain_size;
  //gl64_t *device_src;
  gl64_t *device_dst;
  //cudaMalloc((void**)&device_src, domain_size * ncols * sizeof(gl64_t));
  cudaMalloc((void**)&device_dst, domain_size * ncols * sizeof(gl64_t));
  cudaMemcpy(device_dst, src, domain_size * ncols * sizeof(gl64_t), cudaMemcpyHostToDevice);
  reverse_permutation<<<1, domain_size>>>(device_dst, log_domain_size, ncols);


  cudaMemcpy(dst, device_dst, domain_size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToHost);
  printf("\nrp:\n");
  printf("[");
  for (uint j = 0; j < (1<<log_domain_size) * ncols; j++)
  {
    printf("%lu, ", *((uint64_t *)&dst[j]));
  }
  printf("]\n");

  dim3 blockIdx = dim3(domain_size/2);
  for (uint32_t i = 0; i < log_domain_size; i++) {
    br_ntt_group<<<1, 1, 1>>>(device_dst, i, domain_size, log_domain_size, ncols);
  }

  cudaMemcpy(dst, device_dst, domain_size * ncols * sizeof(gl64_t), cudaMemcpyDeviceToHost);
  cudaFree(device_dst);
}

int main() {
  uint32_t log_domain_size = 4;
  uint32_t ncols = 1;
  uint64_t *a = (uint64_t *)malloc((1<<log_domain_size) * sizeof(uint64_t));
  uint64_t *b = (uint64_t *)malloc((1<<log_domain_size) * sizeof(uint64_t));
  for (uint64_t i = 0; i < 2; i++)
  {
    for (uint64_t j = 0; j < ncols; j++)
    {
      a[i * ncols + j] = 1+j;
    }
  }

  for (uint64_t i = 2; i < (1<<log_domain_size); i++)
  {
    for (uint64_t j = 0; j < ncols; j++)
    {
      a[i * ncols + j] = a[ncols * (i - 1) + j] + a[ncols * (i - 2) + j];
    }
  }

  printf("\ninputs:\n");
  printf("[");
  for (uint j = 0; j < (1<<log_domain_size) * ncols; j++)
  {
    printf("%lu, ", a[j]);
  }
  printf("]\n");

  init_twiddle_factors<<<1,1>>>(log_domain_size);

  NTT_cuda((gl64_t *)b, (gl64_t *)a, log_domain_size, ncols);

  printf("\noutputs:\n");
  printf("[");
  for (uint j = 0; j < (1<<log_domain_size) * ncols; j++)
  {
    printf("%lu, ", b[j]);
  }
  printf("]\n");

}