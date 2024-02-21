#include <gtest/gtest.h>
#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"

TEST(GOLDILOCKS_TEST, avx_op)
{
  Goldilocks::Element *a = (Goldilocks::Element *)malloc(3 * sizeof(Goldilocks::Element));
  a[0] = Goldilocks::fromU64(0X5587AD00B6DDF0CB);
  a[1] = Goldilocks::fromU64(0X279949E14530C250);
  a[2] = Goldilocks::fromU64(0x2F8E22C794677751);

  __m256i st0;
  Goldilocks::load_avx(st0, a);
  const Goldilocks::Element* C_small = &(PoseidonGoldilocksConstants::C[0]);
  __m256i c0;
  Goldilocks::load_avx(c0, &(C_small[0]));

  Goldilocks::add_avx_b_small(st0, st0, c0);

  Goldilocks::store_avx(a, st0);

  for (uint32_t i = 0; i<3; i++) {
    printf("%llu\n", Goldilocks::toU64(a[i]));
  }

  Goldilocks::square_avx(st0, st0);
  Goldilocks::store_avx(a, st0);

  for (uint32_t i = 0; i<3; i++) {
    printf("%llu\n", Goldilocks::toU64(a[i]));
  }

}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
