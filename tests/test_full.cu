#include <gtest/gtest.h>
#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../utils/timer.hpp"

TEST(GOLDILOCKS_TEST, avx_op)
{
  const uint32_t N = 4;
  Goldilocks::Element *a = (Goldilocks::Element *)malloc(N * sizeof(Goldilocks::Element));
  a[0] = Goldilocks::fromU64(0X5587AD00B6DDF0CB);
  a[1] = Goldilocks::fromU64(0X279949E14530C250);
  a[2] = Goldilocks::fromU64(0x2F8E22C794677751);
  a[3] = Goldilocks::fromU64(0X8EC2B67AFB6B87ED);

  __m256i st0;
  Goldilocks::load_avx(st0, a);
  const Goldilocks::Element* C_small = &(PoseidonGoldilocksConstants::C[0]);
  __m256i c0;
  Goldilocks::load_avx(c0, &(C_small[0]));

  Goldilocks::add_avx_b_small(st0, st0, c0);

  Goldilocks::store_avx(a, st0);

  for (uint32_t i = 0; i<N; i++) {
    printf("%lu\n", Goldilocks::toU64(a[i]));
  }

  Goldilocks::square_avx(st0, st0);
  Goldilocks::store_avx(a, st0);

  for (uint32_t i = 0; i<N; i++) {
    printf("%lu\n", Goldilocks::toU64(a[i]));
  }

  Goldilocks::Element *b = (Goldilocks::Element *)malloc(2 * N * sizeof(Goldilocks::Element));

  std::memcpy(b, a, N * sizeof(Goldilocks::Element));
  std::memcpy(b+N, a, N * sizeof(Goldilocks::Element));

  PoseidonGoldilocks::linear_hash(a, b, N * 2);

  for (uint32_t i = 0; i<N; i++) {
    printf("%lu\n", Goldilocks::toU64(a[i]));
  }
}

#define FFT_SIZE (1 << 23)
#define BLOWUP_FACTOR 1
#define NUM_COLUMNS 751

#ifdef __USE_CUDA__
TEST(GOLDILOCKS_TEST, full_gpu)
{
  Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
  Goldilocks::Element *b = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
  Goldilocks::Element *c = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));

  NTT_Goldilocks ntt(FFT_SIZE);

  for (uint i = 0; i < 2; i++)
  {
    for (uint j = 0; j < NUM_COLUMNS; j++)
    {
      Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
    }
  }

  for (uint64_t i = 2; i < FFT_SIZE; i++)
  {
    for (uint j = 0; j < NUM_COLUMNS; j++)
    {
      a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
    }
  }

  TimerStart(LDE_MerkleTree_MultiGPU_v3);
  ntt.LDE_MerkleTree_MultiGPU_v3(b, a, FFT_SIZE, FFT_SIZE<<BLOWUP_FACTOR, NUM_COLUMNS, c);
  TimerStopAndLog(LDE_MerkleTree_MultiGPU_v3);

  printf("dst:\n");
  for (uint64_t i = 0; i < 4; i++) {
    printf("%lu\n", Goldilocks::toU64(b[i]));
    printf("%lu\n", Goldilocks::toU64(b[(uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS - 4 -i]));
  }
  printf("buffer:\n");
  for (uint64_t i = 0; i < 4; i++) {
    printf("%lu\n", Goldilocks::toU64(c[i]));
    printf("%lu\n", Goldilocks::toU64(c[(uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS - 4 -i]));
  }

  uint64_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  printf("free_mem: %lu, total_mem: %lu\n", free_mem, total_mem);

  free(a);
  free(b);
  free(c);
}

TEST(GOLDILOCKS_TEST, full_um)
{
  Goldilocks::Element *a;
  Goldilocks::Element *b;
  Goldilocks::Element *c;
  cudaMallocManaged(&a, (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
  cudaMallocManaged(&b, (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
  cudaMallocManaged(&c, (uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));

  NTT_Goldilocks ntt(FFT_SIZE);

  for (uint i = 0; i < 2; i++)
  {
    for (uint j = 0; j < NUM_COLUMNS; j++)
    {
      Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
    }
  }

  for (uint64_t i = 2; i < FFT_SIZE; i++)
  {
    for (uint j = 0; j < NUM_COLUMNS; j++)
    {
      a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
    }
  }

  TimerStart(LDE_MerkleTree_MultiGPU_v3_um);
  ntt.LDE_MerkleTree_MultiGPU_v3(b, a, FFT_SIZE, FFT_SIZE<<BLOWUP_FACTOR, NUM_COLUMNS, c);
  TimerStopAndLog(LDE_MerkleTree_MultiGPU_v3_um);

  printf("dst:\n");
  for (uint64_t i = 0; i < 4; i++) {
    printf("%lu\n", Goldilocks::toU64(b[i]));
    printf("%lu\n", Goldilocks::toU64(b[(uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS - 4 -i]));
  }
  printf("buffer:\n");
  for (uint64_t i = 0; i < 4; i++) {
    printf("%lu\n", Goldilocks::toU64(c[i]));
    printf("%lu\n", Goldilocks::toU64(c[(uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS - 4 -i]));
  }

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

TEST(GOLDILOCKS_TEST, full_cpu)
{
  Goldilocks::Element *a = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
  Goldilocks::Element *b = (Goldilocks::Element *)malloc((uint64_t)(FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));

  NTT_Goldilocks ntt(FFT_SIZE);

  for (uint i = 0; i < 2; i++)
  {
    for (uint j = 0; j < NUM_COLUMNS; j++)
    {
      Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
    }
  }

  for (uint64_t i = 2; i < FFT_SIZE; i++)
  {
    for (uint j = 0; j < NUM_COLUMNS; j++)
    {
      a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
    }
  }

  TimerStart(LDE_MerkleTree_CPU);
  ntt.LDE_MerkleTree_CPU(b, a, FFT_SIZE, FFT_SIZE<<BLOWUP_FACTOR, NUM_COLUMNS);
  TimerStopAndLog(LDE_MerkleTree_CPU);

  free(a);
  free(b);
}
#endif

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
