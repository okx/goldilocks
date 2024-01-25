#include "../src/goldilocks_cubic_extension.cuh"
#include "test_goldilocks_cubic_extension.hpp"
#include <assert.h>

#define ASSERT_EQ(a,b) assert((a) == (b))

__global__ void test_goldilocks3_kernel() {
    uint64_t a[3] = {1, 1, 1};
    int32_t b[3] = {1, 1, 1};
    uint64_t d[3] = {1 + GOLDILOCKS_PRIME, 1 + GOLDILOCKS_PRIME, 1 + GOLDILOCKS_PRIME};

    Goldilocks3GPU::Element ina1;
    Goldilocks3GPU::Element ina2;
    Goldilocks3GPU::Element ina3;

    Goldilocks3GPU::fromU64(ina1, a);
    Goldilocks3GPU::fromS32(ina2, b);
    Goldilocks3GPU::fromU64(ina3, d);

    uint64_t ina1_res[3];
    uint64_t ina2_res[3];
    uint64_t ina3_res[3];

    Goldilocks3GPU::toU64(ina1_res, ina1);
    Goldilocks3GPU::toU64(ina2_res, ina2);
    Goldilocks3GPU::toU64(ina3_res, ina3);

    ASSERT_EQ(ina1_res[0], a[0]);
    ASSERT_EQ(ina1_res[1], a[1]);
    ASSERT_EQ(ina1_res[2], a[2]);

    ASSERT_EQ(ina2_res[0], a[0]);
    ASSERT_EQ(ina2_res[1], a[1]);
    ASSERT_EQ(ina2_res[2], a[2]);

    ASSERT_EQ(ina3_res[0], a[0]);
    ASSERT_EQ(ina3_res[1], a[1]);
    ASSERT_EQ(ina3_res[2], a[2]);
}

void test_goldilocks3_gpu() {
    test_goldilocks3_kernel<<<1, 32>>>();
}
