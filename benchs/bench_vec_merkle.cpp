#if defined(__USE_SVE__)

#include "../src/goldilocks_base_field_sve.hpp"
#define VEC sve
typedef svuint64_t vectype_t;

#elif defined(__USE_AVX__)
#if defined(__AVX512__)

#include "../src/goldilocks_base_field_avx512.hpp"
#define VEC avx512
typedef __m512i vectype_t;

#else

#include "../src/goldilocks_base_field_avx.hpp"
#define VEC avx
typedef __m256i vectype_t;

#endif  // __AVX512__

#endif  // __USE

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"

#include <sys/time.h>
#include <time.h>
#include <stdlib.h>

Goldilocks::Element *random_data(size_t total_elem)
{
    Goldilocks::Element *data = (Goldilocks::Element *)malloc(total_elem * sizeof(Goldilocks::Element));
    assert(data != NULL);
    srand(0);

#pragma omp parallel for
    for (size_t i = 0; i < total_elem; i++)
    {
        data[i] = Goldilocks::fromS32(rand());
    }
    return data;
}

Goldilocks::Element *const_data(size_t total_elem)
{
    Goldilocks::Element *data = (Goldilocks::Element *)malloc(total_elem * sizeof(Goldilocks::Element));
    assert(data != NULL);
    srand(0);

#pragma omp parallel for
    for (size_t i = 0; i < total_elem; i++)
    {
        data[i] = Goldilocks::fromU64(i * 2147483647);
    }
    return data;
}

void run()
{
    uint64_t ncols = 8;
    uint64_t nrows = (1 << 22);

    Goldilocks::Element* leaves = const_data(ncols * nrows);

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(nrows);
    Goldilocks::Element *tree = (Goldilocks::Element *)malloc(numElementsTree * sizeof(Goldilocks::Element));
    assert(tree != NULL);

    struct timeval start, end;
    gettimeofday(&start, NULL);
#if defined(__USE_SVE__)
    PoseidonGoldilocks::merkletree_sve(tree, leaves, ncols, nrows);
#elif defined(__USE_AVX__)
#ifdef __AVX512__
    PoseidonGoldilocks::merkletree_avx512(tree, leaves, ncols, nrows);
#else
    PoseidonGoldilocks::merkletree_avx(tree, leaves, ncols, nrows);
#endif  // __AVX512__
#else
    PoseidonGoldilocks::merkletree(tree, leaves, ncols, nrows);
#endif  // __USE_AVX__
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Merkle tree building time: %lu ms\n", t / 1000);

    printf("Root is %lx\n", tree[0].fe);

    free(leaves);
    free(tree);
}

int main()
{
    run();
    return 0;
}
