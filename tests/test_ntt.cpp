#include "../src/goldilocks_base_field.hpp"
#include "../src/ntt_goldilocks.hpp"

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
            printf("Data are different at index %lu\n", i);
            return 0;
        }
    }
    printf("Data is the same.\n");
    return 1;
}

void test(char* path, int testId) {
    printf("Running for test id %d\n", testId);

    struct timeval start, end;
    
    char ifilename[64];
    char ofilename[64];
    sprintf(ifilename, "%s/ntt_src-0.bin", path);
    // sprintf(ofilename, "%s/ntt_src-1.bin", path);
    sprintf(ofilename, "%s/leaves-00.bin", path);

    uint64_t iparams[6] = {0};
    // uint64_t oparams[6] = {0};
    uint64_t ine, one;    
    Goldilocks::Element * idata = read_file(ifilename, &ine, iparams);
    printf("Number of input elements %lu\n", ine);
    Goldilocks::Element *odata = (Goldilocks::Element*)malloc(2 * ine * sizeof(Goldilocks::Element));
    assert(odata != NULL);
    Goldilocks::Element *tmp = (Goldilocks::Element *)malloc(2 * ine * sizeof(Goldilocks::Element));
    assert(tmp != NULL);
    Goldilocks::Element * gpu_odata = (Goldilocks::Element*)malloc(ine * sizeof(Goldilocks::Element));
    assert(gpu_odata != NULL);

    NTT_Goldilocks ntt(iparams[0]);
#ifdef __USE_CUDA__
    ntt.setUseGPU(true);
#endif
    ntt.computeR(iparams[0]);
    
    gettimeofday(&start, NULL);
#ifdef __USE_CUDA__
    // ntt.INTT_MultiGPU(odata, idata, iparams[0], iparams[1], tmp, iparams[2], true);
    ntt.Extend_MultiGPU(odata, idata, iparams[0], 2*iparams[0], iparams[1]);
#else
    // ntt.INTT(odata, idata, iparams[0], iparams[1], tmp, iparams[2], iparams[3], true);
    ntt.extendPol(odata, idata, 2*iparams[0], iparams[0], iparams[1], tmp, iparams[2], iparams[3]);
#endif
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("INTT CPU time: %lu ms\n", t / 1000);    

/*
    memset(tmp, 0, 2 * ine * sizeof(Goldilocks::Element));
    ntt.setUseGPU(true);
    gettimeofday(&start, NULL);
    ntt.INTT(gpu_odata, idata, iparams[0], iparams[1], tmp, iparams[2], iparams[3], true);
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("INTT GPU time: %lu ms\n", t / 1000);
*/    
    free(idata);
    free(tmp);
   
    one = 2 * ine;
    Goldilocks::Element * orefdata = read_file_v2(ofilename, one);
    printf("Number of output elements %lu\n", one);  
    comp_output(odata, orefdata, one);
    
    free(orefdata);
    free(gpu_odata);
    free(odata);
    
    printf("Test id %d done.\n", testId);
}

void test_random() {
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
    ntt.extendPol(odata, idata, n_ext, n, ncols, tmp);
    // ntt.NTT(odata, idata, n, ncols, tmp, 3, 1, false, true);
    // ntt.INTT(odata, odata, n_ext, ncols, tmp);
    gettimeofday(&end, NULL);
    uint64_t t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on CPU time: %lu ms\n", t / 1000);

    gettimeofday(&start, NULL);
    ntt.Extend_GPU(gpu_odata, idata, n, n_ext, ncols, tmp);
    // ntt.extendPol(gpu_odata, idata, n_ext, n, ncols, tmp);
    // ntt.NTT_GPU(gpu_odata, idata, n, ncols, tmp, 3, false, true);
    // ntt.INTT_GPU(gpu_odata, gpu_odata, n_ext, ncols, tmp);
    gettimeofday(&end, NULL);
    t = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
    printf("Extend on GPU time: %lu ms\n", t / 1000);

    free(idata);
    free(tmp);
   
    printf("Number of output elements %lu\n", one);  
    comp_output(odata, gpu_odata, one);
        
    free(gpu_odata);
    free(odata);
    
    printf("Test done.\n");
}

int main(int argc, char **argv) {
    // test((char*)"/data/workspace/dumi/x1-prover", 0);
    test_random();
}