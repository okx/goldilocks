#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/merklehash_goldilocks.hpp"

#include <sys/time.h>
#include <sys/mman.h>
#include <fcntl.h>

#ifdef TRACING
#include "tracing.hpp"
#endif

// Read binary file
Goldilocks::Element *read_filex(char *fname, size_t total_elem)
{
    FILE *f = fopen(fname, "rb");
    assert(f != NULL);

    Goldilocks::Element *data = (Goldilocks::Element *)malloc(total_elem * sizeof(Goldilocks::Element));
    assert(data != NULL);

    size_t read_size = 4096;
    size_t read_elem = read_size / sizeof(Goldilocks::Element);
    size_t idx = 0;
    while (total_elem >= read_elem)
    {
        size_t n = fread(data + idx, sizeof(Goldilocks::Element), read_elem, f);
        assert(n == read_elem);
        idx += read_elem;
        total_elem -= read_elem;
    }
    if (total_elem > 0)
    {
        size_t n = fread(data + idx, sizeof(Goldilocks::Element), total_elem, f);
        assert(n == total_elem);
    }
    fclose(f);
    return data;
}

Goldilocks::Element *map_file(char *fname, size_t total_elem)
{
    int fd = open(fname, O_RDONLY);
    assert(-1 != fd);
    return (Goldilocks::Element *)mmap(NULL, total_elem * sizeof(Goldilocks::Element), PROT_READ, MAP_PRIVATE, fd, 0);
}

// compare output tree with the reference tree
int comp_trees(Goldilocks::Element *in_tree, Goldilocks::Element *ref_tree, size_t nelem)
{
    for (size_t i = 0; i < nelem; i++)
    {
        if (in_tree[i].fe != ref_tree[i].fe)
        {
            printf("Trees are different at index %lu\n", i);
            return 0;
        }
    }
    return 1;
}

// Dimensions of input leaves data structure (24 inputs, the data is in files leaves-<index>.bin, 0 <= index < 24, the reference output tree is in files tree-<index>.bin)
const uint64_t ROWS[24] = {16777216, 16777216, 16777216, 16777216, 524288, 16384, 1024, 64, 4194304, 4194304, 4194304, 4194304, 262144, 16384, 1024, 64, 1048576, 1048576, 1048576, 1048576, 65536, 4096, 512, 64};
const uint64_t COLS[24] = {665, 128, 371, 6, 96, 96, 48, 48, 18, 0, 78, 12, 48, 48, 48, 48, 18, 0, 39, 21, 48, 48, 24, 24};

void run_test(int testId, char *path)
{
    printf("Running for tree id %d\n", testId);

    struct timeval start, end;

    uint64_t ncols = COLS[testId];
    uint64_t nrows = ROWS[testId];
    char lfilename[64];
    char tfilename[64];
    sprintf(lfilename, "%s/leaves-%02d.bin", path, testId);
    sprintf(tfilename, "%s/tree-%02d.bin", path, testId);

    uint64_t numElementsTree = MerklehashGoldilocks::getTreeNumElements(nrows);
    Goldilocks::Element *tree = (Goldilocks::Element *)malloc(numElementsTree * sizeof(Goldilocks::Element));
    assert(tree != NULL);

    Goldilocks::Element *leaves = map_file((char *)lfilename, ncols * nrows);

    gettimeofday(&start, NULL);
#ifdef __USE_CUDA__
    PoseidonGoldilocks::merkletree_cuda(tree, leaves, ncols, nrows);
#elif defined(__USE_SVE__)
    PoseidonGoldilocks::merkletree_sve(tree, leaves, ncols, nrows);
#else
    PoseidonGoldilocks::merkletree_avx(tree, leaves, ncols, nrows);
#endif
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Merkle tree building time: %lu ms\n", t / 1000);
    
    munmap(leaves, ncols * nrows * sizeof(Goldilocks::Element));

    Goldilocks::Element *ref_tree = map_file((char *)tfilename, numElementsTree);
    printf("Trees are the same: %d\n", comp_trees(tree, ref_tree, numElementsTree));

    free(tree);
    munmap(ref_tree, numElementsTree * sizeof(Goldilocks::Element));
}
int main(int argc, char **argv)
{
    if (argc <= 1)
    {
        printf("Usage: ./%s <path-to-input-files>\n", argv[0]);
        return 1;
    }

#ifdef TRACING
    open_write_files();
#endif

    if (argc > 2)
    {
        int tid = atoi(argv[2]);
        if (tid < 0 || tid > 23)
        {
            printf("Test id needs to be >= 0 and <= 23!\n");
            return 1;
        }
        run_test(tid, argv[1]);
        return 0;
    }
    else
    {
        for (int i = 0; i < 24; i++)
        {
            run_test(i, argv[1]);
        }
    }

#ifdef TRACING
    close_files();
#endif
    return 0;
}