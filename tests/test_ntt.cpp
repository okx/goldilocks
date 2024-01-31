#include "../src/goldilocks_base_field.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/poseidon_goldilocks.hpp"

#include <sys/time.h>
#include <sys/mman.h>
#include <time.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

// Read binary file with the following format
// - the first 6 elements (each of 8 bytes) are: nrows (or size), ncols, nphase, nblock, inverse, extend
// - the next nrows * ncols elements (each of 8 bytes) are the actual data
Goldilocks::Element *read_file(char *fname, uint64_t* n_elem, uint64_t* params)
{
    FILE *f = fopen(fname, "rb");
    assert(f != NULL);
    assert(6 == fread(params, sizeof(uint64_t), 6, f));
    *n_elem = params[0] * params[1];
    uint64_t total_elem = *n_elem;

    Goldilocks::Element *data = (Goldilocks::Element*)malloc(total_elem * sizeof( Goldilocks::Element));
    assert(data != NULL);
    size_t read_size = 256 * 1024 * 1024;
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

// Read binary file with data only by providing the total number of elements
Goldilocks::Element *read_file_v2(char *fname, size_t total_elem)
{
    FILE *f = fopen(fname, "rb");
    assert(f != NULL);

    Goldilocks::Element *data = (Goldilocks::Element *)malloc(total_elem * sizeof(Goldilocks::Element));
    assert(data != NULL);

    size_t read_size = 256 * 1024 * 1024;
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


// compare output with the reference output
int comp_output(Goldilocks::Element *in, Goldilocks::Element *ref, size_t nelem)
{
// #pragma omp parallel for
    for (size_t i = 0; i < nelem; i++)
    {
        if (in[i].fe != ref[i].fe)
        {
            printf("!!! Data are different at index %lu\n", i);
            return 0;
        }
    }
    printf("Data is the same.\n");
    return 1;
}

int test(char* path, int testId) {
    printf("Running for test id %d\n", testId);

    struct timeval start, end;

    char ifilename[64];
    char ofilename[64];
    sprintf(ifilename, "%s/ntt_src-0.bin", path);
    // sprintf(ofilename, "%s/ntt_src-1.bin", path);
    // sprintf(ofilename, "%s/leaves-00.bin", path);
    sprintf(ofilename, "%s/tree-00.bin", path);

    uint64_t iparams[6] = {0};
    // uint64_t oparams[6] = {0};
    uint64_t ine, one;
    Goldilocks::Element * idata = read_file(ifilename, &ine, iparams);
    printf("Number of input elements %lu\n", ine);
    uint64_t tree_size = (2 * iparams[0] - 1) * 4;
    printf("Number of tree elements %lu\n", tree_size);
    Goldilocks::Element *tree1 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree1 != NULL);
    Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(2 * ine * sizeof(Goldilocks::Element));
    assert(tmp != NULL);
    Goldilocks::Element * tree2 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree2 != NULL);

    NTT_Goldilocks ntt(iparams[0]);
#ifdef __USE_CUDA__
    ntt.setUseGPU(true);
#endif
    ntt.computeR(iparams[0]);

    gettimeofday(&start, NULL);    
    ntt.extendPol(tmp, idata, 2*iparams[0], iparams[0], iparams[1], NULL, iparams[2], iparams[3]);
    PoseidonGoldilocks::merkletree_avx(tree1, tmp, iparams[1], 2*iparams[0]);
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("LDE + Merkle Tree on CPU time: %lu ms\n", t / 1000);

    memset(tmp, 0, 2 * ine * sizeof(Goldilocks::Element));
    ntt.setUseGPU(true);
    gettimeofday(&start, NULL);
    ntt.INTT(tmp, idata, iparams[0], iparams[1], NULL, 3, 1, true);
    ntt.NTT_BatchGPU(tree2, tmp, 2*iparams[0], iparams[1], 80, NULL, 3, false, false, true);
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("LDE + Merkle Tree on GPU time: %lu ms\n", t / 1000);

    free(idata);
    free(tmp);

    one = 2 * ine;
    Goldilocks::Element * orefdata = read_file_v2(ofilename, one);
    printf("Number of output elements %lu\n", one);
    int ret = comp_output(tree1, tree2, tree_size);

    free(orefdata);
    free(tree1);
    free(tree2);

    printf("Test id %d done.\n\n", testId);
    return ret;
}

int test_random() {
    printf("Running test with random data...\n");

    struct timeval start, end;

    uint64_t ncols = 32;
    uint64_t ine = 1 << 23;
    uint64_t one = 1 << 24; // blowup factor 2
    uint64_t n = ine / ncols;
    uint64_t n_ext = one / ncols;

    Goldilocks::Element * idata = random_data(ine);
    printf("Number of input elements %lu\n", ine);
    Goldilocks::Element *odata = (Goldilocks::Element*)malloc(one * sizeof(Goldilocks::Element));
    assert(odata != NULL);
    Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(one * sizeof(Goldilocks::Element));
    assert(tmp != NULL);
    Goldilocks::Element * gpu_odata = (Goldilocks::Element*)malloc(one * sizeof(Goldilocks::Element));
    assert(gpu_odata != NULL);

    NTT_Goldilocks ntt(n_ext);
#ifdef __USE_CUDA__
    ntt.setUseGPU(true);
#endif
    ntt.computeR(n_ext);

    gettimeofday(&start, NULL);
    // ntt.extendPol(odata, idata, n_ext, n, ncols, tmp);
    ntt.NTT(odata, idata, n, ncols, tmp, 3, 1, false, true);
    // ntt.INTT(odata, odata, n_ext, ncols, tmp);
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on CPU time: %lu ms\n", t / 1000);

    gettimeofday(&start, NULL);
    // ntt.Extend_GPU(gpu_odata, idata, n, n_ext, ncols, tmp);
    // ntt.extendPol(gpu_odata, idata, n_ext, n, ncols, tmp);
    ntt.NTT_BatchGPU(gpu_odata, idata, n, ncols, 24, tmp, 3, false, true);
    // ntt.NTT_GPU(gpu_odata, idata, n, ncols, tmp, 3, false, true);
    // ntt.INTT_GPU(gpu_odata, gpu_odata, n_ext, ncols, tmp);
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on GPU time: %lu ms\n", t / 1000);

    free(idata);
    free(tmp);

    printf("Number of output elements %lu\n", one);
    int ret = comp_output(odata, gpu_odata, one);

    free(gpu_odata);
    free(odata);

    printf("Test done.\n\n");
    return ret;
}

int test_ntt_partial_hash_random() {
    printf("Running NTT + Merkle Tree with partial hash test with random data...\n");

    struct timeval start, end;

    uint64_t ncols = 32;
    uint64_t ine = 1 << 23;
    uint64_t n = ine / ncols;
    uint64_t n_tree_elem = 2 * n - 1;
    uint64_t tree_size = n_tree_elem * 4;

    Goldilocks::Element * idata = random_data(ine);
    printf("Number of input elements %lu\n", ine);
    Goldilocks::Element *tree1 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree1 != NULL);
    Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(ine * sizeof(Goldilocks::Element));
    assert(tmp != NULL);
    Goldilocks::Element * tree2 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree2 != NULL);

    NTT_Goldilocks ntt(n);
#ifdef __USE_CUDA__
    ntt.setUseGPU(true);
#endif
    ntt.computeR(n);

    gettimeofday(&start, NULL);
    ntt.NTT(tmp, idata, n, ncols);
    PoseidonGoldilocks::merkletree_avx(tree1, tmp, ncols, n);
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("NTT + Merkle Tree on CPU time: %lu ms\n", t / 1000);

    gettimeofday(&start, NULL);
    ntt.NTT_BatchGPU(tree2, idata, n, ncols, 8, tmp, 3, false, false, true);
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("NTT + Merkle Tree batch on GPU time: %lu ms\n", t / 1000);

    free(idata);
    free(tmp);

    printf("Number of output elements %lu\n", tree_size);
    int ret = comp_output(tree1, tree2, tree_size);

    free(tree1);
    free(tree2);

    printf("Test done.\n\n");
    return ret;
}

int test_lde_no_merkle_random() {
    printf("Running LDE only on random data...\n");

    struct timeval start, end;

    uint64_t ncols = 32;
    uint64_t ine = 1 << 23;
    uint64_t one = 1 << 24; // blowup factor 2
    uint64_t n = ine / ncols;
    uint64_t n_ext = one / ncols;

    Goldilocks::Element * idata = random_data(ine);
    printf("Number of input elements %lu\n", ine);
    Goldilocks::Element *odata1 = (Goldilocks::Element*)malloc(one * sizeof(Goldilocks::Element));
    assert(odata1 != NULL);
    Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(one * sizeof(Goldilocks::Element));
    assert(tmp != NULL);
    Goldilocks::Element * odata2 = (Goldilocks::Element*)malloc(one * sizeof(Goldilocks::Element));
    assert(odata2 != NULL);

    NTT_Goldilocks ntt(n_ext);
#ifdef __USE_CUDA__
    ntt.setUseGPU(true);
#endif
    ntt.computeR(n_ext);

    gettimeofday(&start, NULL);
    ntt.LDE_MerkleTree_CPU(odata1, idata, n, n_ext, ncols, tmp, false); // false means do not build the Merkle Tree
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on CPU time: %lu ms\n", t / 1000);

    gettimeofday(&start, NULL);
    ntt.LDE_MerkleTree_GPU(odata2, idata, n, n_ext, ncols, tmp, false); // false means do not build the Merkle Tree
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on GPU time: %lu ms\n", t / 1000);

    free(idata);
    free(tmp);

    printf("Number of output elements %lu\n", one);
    int ret = comp_output(odata1, odata2, one);

    free(odata1);
    free(odata2);

    printf("Test done.\n\n");
    return ret;
}

int test_lde_merkle_random() {
    printf("Running LDE + Merkle Tree on random data...\n");

    struct timeval start, end;

    uint64_t ncols = 32;
    uint64_t ine = 1 << 23;
    uint64_t one = 1 << 24; // blowup factor 2
    uint64_t n = ine / ncols;
    uint64_t n_ext = one / ncols;
    uint64_t n_tree_elem = 2 * n_ext - 1;
    uint64_t tree_size = n_tree_elem * 4;

    Goldilocks::Element * idata = random_data(ine);
    printf("Number of input elements %lu\n", ine);
    printf("Tree size %lu\n", tree_size);
    Goldilocks::Element *tree1 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree1 != NULL);
    Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(one * sizeof(Goldilocks::Element));
    assert(tmp != NULL);
    Goldilocks::Element * tree2 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree2 != NULL);

    NTT_Goldilocks ntt(n_ext);
#ifdef __USE_CUDA__
    ntt.setUseGPU(true);
#endif
    ntt.computeR(n_ext);

    gettimeofday(&start, NULL);
    ntt.LDE_MerkleTree_CPU(tree1, idata, n, n_ext, ncols, tmp);
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on CPU time: %lu ms\n", t / 1000);

    gettimeofday(&start, NULL);
    ntt.LDE_MerkleTree_GPU(tree2, idata, n, n_ext, ncols, tmp);
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on GPU time: %lu ms\n", t / 1000);

    free(idata);
    free(tmp);

    printf("Number of output elements %lu\n", tree_size);
    int ret = comp_output(tree1, tree2, tree_size);

    free(tree1);
    free(tree2);

    printf("Test done.\n\n");
    return ret;
}

int main(int argc, char **argv) {
    // assert(1 == test((char*)"/data/workspace/dumi/x1-prover", 0));

    assert(1 == test_random());

    assert(1 == test_lde_no_merkle_random());

    assert(1 == test_lde_merkle_random());

    assert(1 == test_ntt_partial_hash_random());
}