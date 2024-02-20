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
#include <math.h>
#include <errno.h>

// Read binary file with the following format
// - the first 2 elements (each of 8 bytes) are: nrows (or size), and ncols
// - the next nrows * ncols elements (each of 8 bytes) are the actual data
Goldilocks::Element *read_file(char *fname, uint64_t* n_elem, uint64_t* params)
{
    FILE *f = fopen(fname, "rb");
    assert(f != NULL);
    assert(2 == fread(params, sizeof(uint64_t), 2, f));
    *n_elem = params[0] * params[1];
    uint64_t total_elem = *n_elem;

    Goldilocks::Element *data = (Goldilocks::Element*)malloc(total_elem * sizeof( Goldilocks::Element));
    assert(data != NULL);
    size_t read_size = (1 << 28);
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

// Read binary file with the following format
// - the first 6 elements (each of 8 bytes) are: nrows (or size), ncols, nphase, nblock, inverse, extend
// - the next nrows * ncols elements (each of 8 bytes) are the actual data
Goldilocks::Element *read_file_v1(char *fname, uint64_t* n_elem, uint64_t* params)
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

void read_file_v3(char *fname, uint64_t* n_elem, uint64_t* params)
{
    FILE *f = fopen(fname, "rb");
    assert(f != NULL);
    assert(6 == fread(params, sizeof(uint64_t), 6, f));
    *n_elem = params[0] * params[1];
    fclose(f);
}

int read_file_v4(char *fname, uint64_t* n_elem, uint64_t* params)
{
    FILE *f = fopen(fname, "rb");
    if (!f)
    {
        return 0;
    }
    assert(2 == fread(params, sizeof(uint64_t), 2, f));
    *n_elem = params[0] * params[1];
    fclose(f);
    return 1;
}

Goldilocks::Element *map_file(char *fname, size_t total_elem, int *fd)
{
    *fd = open(fname, O_RDONLY);
    assert(-1 != *fd);
    void *ret = mmap(NULL, total_elem * sizeof(Goldilocks::Element), PROT_READ, MAP_PRIVATE, *fd, 0);
    printf("Errno: %d\n", errno);
    assert(-1 != (int64_t)ret);
    return (Goldilocks::Element *)ret;
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
    printf("Compare %lu elements ...\n", nelem);
// #pragma omp parallel for
    for (size_t i = 0; i < nelem; i++)
    {
        if (in[i].fe != ref[i].fe)
        {
            printf("!!! Data are different at index %lu\n", i);
            printf("%lu vs. %lu\n", in[i].fe, ref[i].fe);
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
    sprintf(ifilename, "%s/pols-%d.bin", path, testId);

    uint64_t iparams[6] = {0};
    // uint64_t oparams[6] = {0};
    uint64_t ine, one;
    // Goldilocks::Element * idata = read_file(ifilename, &ine, iparams);
    if (read_file_v4(ifilename, &ine, iparams) == 0)
    {
        printf("File not found %s.\n\n", ifilename);
        return 1;
    }
    // iparams[1] = 640;
    ine = iparams[0] * iparams[1];
    if (ine == 0)
    {
        printf("0 input elements. Return 1.\n\n");
        return 1;
    }
    one = 2 * ine;
    int fd;
    Goldilocks::Element * idata1 = map_file(ifilename, ine + 2, &fd);
    Goldilocks::Element * idata = idata1 + 2;
    printf("Number of input elements %lu\n", ine);
    uint64_t tree_size = (4 * iparams[0] - 1) * 4;  // 4 * n -> n_ext
    printf("Number of tree elements %lu\n", tree_size);
    Goldilocks::Element *tree1 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree1 != NULL);
    Goldilocks::Element *out1 = (Goldilocks::Element *)malloc(one * sizeof(Goldilocks::Element));
    assert(out1 != NULL);
    Goldilocks::Element * tree2 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree2 != NULL);
     Goldilocks::Element *out2 = (Goldilocks::Element *)malloc(one * sizeof(Goldilocks::Element));
    assert(out2 != NULL);

    NTT_Goldilocks ntt(2 * iparams[0]);
#ifdef __USE_CUDA__
    ntt.setUseGPU(true);
#endif
    // IMPORTANT: it needs to be done for the original size, not extended
    ntt.computeR(iparams[0]);

    // warmup
    // ntt.NTT(tmp, idata, iparams[0], iparams[1]);
    printf("Start LDE on CPU ...\n");
    gettimeofday(&start, NULL);
    // ntt.extendPol(tmp, idata, 2*iparams[0], iparams[0], iparams[1], NULL, iparams[2], iparams[3]);
    // PoseidonGoldilocks::merkletree_avx(tree1, tmp, iparams[1], 2*iparams[0]);
    ntt.LDE_MerkleTree_CPU(tree1, idata, iparams[0], 2 * iparams[0], iparams[1], out1);
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("LDE + Merkle Tree on CPU time: %lu ms\n", t / 1000);

    // memset(tmp, 0, 2 * ine * sizeof(Goldilocks::Element));
    ntt.setUseGPU(true);
    gettimeofday(&start, NULL);
    // ntt.INTT(tmp, idata, iparams[0], iparams[1], NULL, 3, 1, true);
    // ntt.NTT_BatchGPU(tree2, tmp, 2*iparams[0], iparams[1], 80, NULL, 3, false, false, true);
    // ntt.LDE_MerkleTree_GPU_v3(tree2, idata, iparams[0], 2 * iparams[0], iparams[1]);
    if (iparams[1] > 100)
    {
        ntt.LDE_MerkleTree_MultiGPU_v3(tree2, idata, iparams[0], 2 * iparams[0], iparams[1], out2);
    }
    else
    {
        ntt.LDE_MerkleTree_GPU_v3(tree2, idata, iparams[0], 2 * iparams[0], iparams[1], out2);
    }
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("LDE + Merkle Tree on GPU time: %lu ms\n", t / 1000);

    // free(idata);
    assert(-1 != munmap(idata1, (ine + 2) * sizeof(Goldilocks::Element)));
    close(fd);

    // here we make sure data is in canonical form
    /*
#pragma omp parallel for
    for (size_t i = 0; i < ine; i++)
    {
        out2[i] = Goldilocks::fromU64(Goldilocks::toU64(out2[i]));
    }
    */

    int ret = comp_output(out1, out2, one);

    free(out1);
    free(out2);

/*
    one = 2 * ine;
    Goldilocks::Element * orefdata = read_file_v2(ofilename, one);
    printf("Number of output elements %lu\n", one);
    int ret = comp_output(tree1, tree2, tree_size);
    free(orefdata);
*/

    ret = comp_output(tree1, tree2, tree_size);
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

    uint64_t ncols = 665;
    uint64_t ine = ncols * (1 << 20);
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
    ntt.NTT_BatchGPU(tree2, idata, n, ncols, 88, tmp, 3, false, false, true);
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

    uint64_t ncols = 84;
    uint64_t ine = ncols * (1 << 23);
    uint64_t one = ine << 1; // blowup factor 2
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
    ntt.computeR(n);

    gettimeofday(&start, NULL);
    ntt.LDE_MerkleTree_CPU(odata1, idata, n, n_ext, ncols, tmp, false); // false means do not build the Merkle Tree
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on CPU time: %lu ms\n", t / 1000);

    gettimeofday(&start, NULL);
    // ntt.LDE_MerkleTree_GPU(odata2, idata, n, n_ext, ncols, tmp, false); // false means do not build the Merkle Tree
    // ntt.LDE_MerkleTree_MultiGPU(odata2, idata, n, n_ext, ncols, tmp, 3, false);
    ntt.LDE_MerkleTree_GPU_v3(odata2, idata, n, n_ext, ncols, tmp, false); // false means do not build the Merkle Tree
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

    uint64_t ncols = 84;
    uint64_t ine = ncols * (1 << 23);
    uint64_t one = 2 * ine; // blowup factor 2
    uint64_t n = ine / ncols;
    uint64_t n_ext = one / ncols;
    uint64_t n_tree_elem = 2 * n_ext - 1;
    uint64_t tree_size = n_tree_elem * 4;

    Goldilocks::Element * idata = random_data(ine);
    printf("Number of input elements %lu\n", ine);
    printf("Tree size %lu\n", tree_size);
    Goldilocks::Element *tree1 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree1 != NULL);
    Goldilocks::Element *tmp1 = (Goldilocks::Element *)malloc(one * sizeof(Goldilocks::Element));
    assert(tmp1 != NULL);
    Goldilocks::Element *tmp2 = (Goldilocks::Element *)malloc(one * sizeof(Goldilocks::Element));
    assert(tmp2 != NULL);
    Goldilocks::Element * tree2 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree2 != NULL);

    NTT_Goldilocks ntt(n_ext);
#ifdef __USE_CUDA__
    ntt.setUseGPU(true);
#endif
    ntt.computeR(n);

    gettimeofday(&start, NULL);
    ntt.LDE_MerkleTree_CPU(tree1, idata, n, n_ext, ncols, tmp1);
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on CPU time: %lu ms\n", t / 1000);

    gettimeofday(&start, NULL);
    ntt.LDE_MerkleTree_GPU_v3(tree2, idata, n, n_ext, ncols);
    // ntt.LDE_MerkleTree_MultiGPU(tree2, idata, n, n_ext, ncols, tmp2);
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on GPU time: %lu ms\n", t / 1000);

    free(idata);
    free(tmp1);
    free(tmp2);

    printf("Number of output elements %lu\n", tree_size);
    int ret = comp_output(tree1, tree2, tree_size);

    free(tree1);
    free(tree2);

    printf("Test done.\n\n");
    return ret;
}

int test_lde_merkle_batch_random() {
    printf("Running LDE + Merkle Tree Batch GPU on random data...\n");

    struct timeval start, end;

    uint64_t ncols = 64;
    uint64_t ine = ncols * (1 << 23);
    uint64_t one = 2 * ine;
    uint64_t n = ine / ncols;
    uint64_t n_ext = one / ncols;
    uint64_t n_tree_elem = 2 * n_ext - 1;
    uint64_t tree_size = n_tree_elem * 4;

    Goldilocks::Element * idata = random_data(ine);
    printf("Rows %lu and columns %lu\n", n, ncols);
    printf("Tree size %lu (log2 is %lu)\n", tree_size, (uint64_t)log2(tree_size));
    Goldilocks::Element *tree1 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree1 != NULL);
    Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(one * sizeof(Goldilocks::Element));
    assert(tmp != NULL);
    Goldilocks::Element *tmp2 = (Goldilocks::Element *)malloc(one * sizeof(Goldilocks::Element));
    assert(tmp2 != NULL);
    Goldilocks::Element * tree2 = (Goldilocks::Element*)malloc(tree_size * sizeof(Goldilocks::Element));
    assert(tree2 != NULL);

    NTT_Goldilocks ntt(n_ext, 0, n_ext/n);
#ifdef __USE_CUDA__
    ntt.setUseGPU(true);
#endif
    ntt.computeR(n_ext);

    gettimeofday(&start, NULL);
    ntt.extendPol(tmp, idata, n_ext, n, ncols);
    PoseidonGoldilocks::merkletree_avx(tree1, tmp, ncols, n_ext);
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("LDE + Merkle Tree on CPU time: %lu ms\n", t / 1000);

    gettimeofday(&start, NULL);
    // ntt.LDE_BatchGPU(tmp2, idata, n, n_ext, ncols, 8, false);
    ntt.LDE_MerkleTree_BatchGPU(tree2, idata, n, n_ext, ncols, 8);
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("LDE + Merkle Tree on GPU time: %lu ms\n", t / 1000);

    printf("Number of output elements %lu\n", tree_size);
    int ret = comp_output(tree1, tree2, tree_size);
    // int ret = comp_output(tmp, tmp2, one);

    free(idata);
    free(tmp);
    free(tmp2);
    free(tree1);
    free(tree2);

    printf("Test done.\n\n");
    return ret;
}

int main(int argc, char **argv) {
    for (int k = 0; k <= 8; k++)
    {
        assert(1 == test((char*)"/data/workspace/dumi/x1-prover", k));
    }

    // assert(1 == test((char*)"/data/workspace/dumi/x1-prover", 0));

    // assert(1 == test_random());

    // assert(1 == test_lde_no_merkle_random());

    // assert(1 == test_lde_merkle_random());

    // assert(1 == test_ntt_partial_hash_random());

    // assert(1 == test_lde_merkle_batch_random());
}