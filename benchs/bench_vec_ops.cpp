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

#endif // __AVX512__

#endif // __USE

#define DELTA   2147483647      // 2^31 - 1

#define PASTER1(x, y) x##_##y
#define EVALUATOR1(x, y) PASTER1(x, y)
#define PASTER2(x, y, z) x##_##y##_##z
#define EVALUATOR2(x, y, z) PASTER2(x, y, z)
#define FUNC1(g, prefix) (g).EVALUATOR1(prefix, VEC)
#define FUNC2(g, prefix, suffix) (g).EVALUATOR2(prefix, VEC, suffix)

#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef __USE_PAPI__
#include "papi.hpp"
#endif

#define MAX_ITER 10000000
// #define MAX_ITER 10

#define REPS 10

FILE *f = NULL;

void print_vectype_t(const vectype_t &a)
{
    uint64_t buffer[4];
#ifdef __USE_SVE__
    svst1_u64(svptrue_b64(), buffer, a);
#else
    _mm256_storeu_si256((__m256i *)buffer, a);
#endif
    // FILE *f = fopen("/dev/null", "w");
    assert(f != NULL);
    for (int i = 3; i >= 0; i--)
    {
        fprintf(f, "%lX ", buffer[i]);
        // printf("%lX ", buffer[i]);
    }
    // printf("\n");
    // fclose(f);
}

void bench_op2(void (*f)(vectype_t &, const vectype_t &))
{
    vectype_t r1, r2, r3, r4, a1, a2, a3, a4;

    struct timeval start, end;

    srand(time(NULL));

#if defined(__USE_SVE__)
    a1 = svdup_u64(DELTA + rand());
    a2 = svdup_u64(DELTA + rand());
    a3 = svdup_u64(DELTA + rand());
    a4 = svdup_u64(DELTA + rand());
#elif defined(__USE_AVX__)
    int64_t r = DELTA + rand();
    a1 = _mm256_set_epi64x(r, r, r, r);
    r = DELTA + rand();
    a2 = _mm256_set_epi64x(r, r, r, r);
    r = DELTA + rand();
    a3 = _mm256_set_epi64x(r, r, r, r);
    r = DELTA + rand();
    a4 = _mm256_set_epi64x(r, r, r, r);
#endif

#ifdef __USE_PAPI__
    papi_init();
#endif
    uint64_t tt = 0;
    gettimeofday(&start, NULL);
    for (int r = 0; r < REPS; r++)
    {
        for (uint64_t i = 0; i < MAX_ITER; i++)
        {
            f(r1, a1);
            f(r2, a2);
            f(r3, a3);
            f(r4, a4);
            f(a1, r1);
            f(a2, r2);
            f(a3, r3);
            f(a4, r4);
        }
    }
    gettimeofday(&end, NULL);
    tt = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
#ifdef __USE_PAPI__
    papi_stop();
#endif

    double at = tt / (1000.0f * REPS);
    printf("Elapsed time: %3.4lf ms\n", at);
    printf("Throughput: %3.4lf Mops/s\n\n", 8 * MAX_ITER / at);

    print_vectype_t(a1);
    print_vectype_t(a2);
    print_vectype_t(a3);
    print_vectype_t(a4);
}

void bench_op3_v1(void (*f)(vectype_t &, const vectype_t &, const vectype_t &))
{
    vectype_t r1, r2, r3, r4, a1, a2, a3, a4, b1, b2, b3, b4;

    struct timeval start, end;

    srand(time(NULL));

#if defined(__USE_SVE__)
    uint64_t r = DELTA + rand();
    a1 = svdup_u64(r);
    a2 = svdup_u64(r);
    a3 = svdup_u64(r);
    a4 = svdup_u64(r);
    b1 = svdup_u64(r);
    b2 = svdup_u64(r);
    b3 = svdup_u64(r);
    b4 = svdup_u64(r);
#elif defined(__USE_AVX__)
    int64_t r = DELTA + rand();
    a1 = _mm256_set_epi64x(r, r, r, r);
    a2 = _mm256_set_epi64x(r, r, r, r);
    a3 = _mm256_set_epi64x(r, r, r, r);
    a4 = _mm256_set_epi64x(r, r, r, r);
    b1 = _mm256_set_epi64x(r, r, r, r);
    b2 = _mm256_set_epi64x(r, r, r, r);
    b3 = _mm256_set_epi64x(r, r, r, r);
    b4 = _mm256_set_epi64x(r, r, r, r);
#endif

#ifdef __USE_PAPI__
    papi_init();
#endif
    uint64_t tt = 0;
    gettimeofday(&start, NULL);
    for (int r = 0; r < REPS; r++)
    {
        for (uint64_t i = 0; i < MAX_ITER; i++)
        {
            f(r1, a1, b1);
            f(r2, a2, b2);
            f(r3, a3, b3);
            f(r4, a4, b4);
            f(a1, b1, r1);
            f(a2, b2, r2);
            f(a3, b3, r3);
            f(a4, b4, r4);
        }
    }
    gettimeofday(&end, NULL);
    tt = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
#ifdef __USE_PAPI__
    papi_stop();
#endif

    double at = tt / (1000.0f * REPS);
    printf("Elapsed time: %3.4lf ms\n", at);
    printf("Throughput: %3.4lf Mops/s\n\n", 8 * MAX_ITER / at);

    print_vectype_t(a1);
    print_vectype_t(a2);
    print_vectype_t(a3);
    print_vectype_t(a4);
}

void bench_op3_v2(void (*f)(vectype_t &, vectype_t &, const vectype_t &))
{
    vectype_t r1, r2, r3, r4, a1, a2, a3, a4, b1, b2, b3, b4;

    struct timeval start, end;

    srand(time(NULL));

#if defined(__USE_SVE__)
    uint64_t r = DELTA + rand();
    a1 = svdup_u64(r);
    a2 = svdup_u64(r);
    a3 = svdup_u64(r);
    a4 = svdup_u64(r);
    b1 = svdup_u64(r);
    b2 = svdup_u64(r);
    b3 = svdup_u64(r);
    b4 = svdup_u64(r);
#elif defined(__USE_AVX__)
    int64_t r = DELTA + rand();
    a1 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a2 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a3 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a4 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b1 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b2 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b3 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b4 = _mm256_set_epi64x(r, r, r, r);
#endif

#ifdef __USE_PAPI__
    papi_init();
#endif
    uint64_t tt = 0;
    gettimeofday(&start, NULL);
    for (int r = 0; r < REPS; r++)
    {
        for (uint64_t i = 0; i < MAX_ITER; i++)
        {
            f(r1, a1, b1);
            f(r2, a2, b2);
            f(r3, a3, b3);
            f(r4, a4, b4);
            f(a1, b1, r1);
            f(a2, b2, r2);
            f(a3, b3, r3);
            f(a4, b4, r4);
        }
    }
    gettimeofday(&end, NULL);
    tt = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
#ifdef __USE_PAPI__
    papi_stop();
#endif

    double at = tt / (1000.0f * REPS);
    printf("Elapsed time: %3.4lf ms\n", at);
    printf("Throughput: %3.4lf Mops/s\n\n", 8 * MAX_ITER / at);

    print_vectype_t(a1);
    print_vectype_t(a2);
    print_vectype_t(a3);
    print_vectype_t(a4);
}

void bench_op4_v1(void (*f)(vectype_t &, vectype_t &, const vectype_t &, const vectype_t &))
{
    vectype_t r11, r12, r21, r22, r31, r32, r41, r42, a1, a2, a3, a4, b1, b2, b3, b4;

    struct timeval start, end;

    srand(time(NULL));

#if defined(__USE_SVE__)
    uint64_t r = DELTA + rand();
    a1 = svdup_u64(r);
    a2 = svdup_u64(r);
    a3 = svdup_u64(r);
    a4 = svdup_u64(r);
    b1 = svdup_u64(r);
    b2 = svdup_u64(r);
    b3 = svdup_u64(r);
    b4 = svdup_u64(r);
#elif defined(__USE_AVX__)
    int64_t r = DELTA + rand();
    a1 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a2 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a3 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a4 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b1 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b2 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b3 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b4 = _mm256_set_epi64x(r, r, r, r);
#endif

#ifdef __USE_PAPI__
    papi_init();
#endif
    uint64_t tt = 0;
    gettimeofday(&start, NULL);
    for (int r = 0; r < REPS; r++)
    {
        for (uint64_t i = 0; i < MAX_ITER; i++)
        {
            f(r11, r12, a1, b1);
            f(r21, r22, a2, b2);
            f(r31, r32, a3, b3);
            f(r41, r42, a4, b4);
            f(a1, b1, r11, r12);
            f(a2, b2, r21, r22);
            f(a3, b3, r31, r32);
            f(a4, b4, r41, r42);
        }
    }
    gettimeofday(&end, NULL);
    tt = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
#ifdef __USE_PAPI__
    papi_stop();
#endif

    double at = tt / (1000.0f * REPS);
    printf("Elapsed time: %3.4lf ms\n", at);
    printf("Throughput: %3.4lf Mops/s\n\n", 8 * MAX_ITER / at);

    print_vectype_t(a1);
    print_vectype_t(a2);
    print_vectype_t(a3);
    print_vectype_t(a4);
}

// Params: res, a0, a1, a2, matrix
void bench_op5_v1(void (*f)(vectype_t &, const vectype_t &, const vectype_t &, const vectype_t &, const Goldilocks::Element[12]))
{
    vectype_t r1, r2, r3, r4, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4;

    const Goldilocks::Element b_8[12] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    struct timeval start, end;

    srand(time(NULL));

#if defined(__USE_SVE__)
    uint64_t r = DELTA + rand();
    a1 = svdup_u64(r);
    a2 = svdup_u64(r);
    a3 = svdup_u64(r);
    a4 = svdup_u64(r);
    b1 = svdup_u64(r);
    b2 = svdup_u64(r);
    b3 = svdup_u64(r);
    b4 = svdup_u64(r);
    c1 = svdup_u64(r);
    c2 = svdup_u64(r);
    c3 = svdup_u64(r);
    c4 = svdup_u64(r);
#elif defined(__USE_AVX__)
    int64_t r = DELTA + rand();
    a1 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a2 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a3 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a4 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b1 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b2 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b3 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b4 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    c1 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    c2 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    c3 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    c4 = _mm256_set_epi64x(r, r, r, r);
#endif

#ifdef __USE_PAPI__
    papi_init();
#endif
    uint64_t tt = 0;
    gettimeofday(&start, NULL);
    for (int r = 0; r < REPS; r++)
    {
        for (uint64_t i = 0; i < MAX_ITER; i++)
        {
            f(r1, a1, b1, c1, b_8);
            f(r2, a2, b2, c2, b_8);
            f(r3, a3, b3, c3, b_8);
            f(r4, a4, b4, c4, b_8);
            f(a1, r1, b1, c1, b_8);
            f(a2, r2, b2, c2, b_8);
            f(a3, r3, b3, c3, b_8);
            f(a4, r4, b4, c4, b_8);
        }
    }
    gettimeofday(&end, NULL);
    tt = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
#ifdef __USE_PAPI__
    papi_stop();
#endif

    double at = tt / (1000.0f * REPS);
    printf("Elapsed time: %3.4lf ms\n", at);
    printf("Throughput: %3.4lf Mops/s\n\n", 8 * MAX_ITER / at);

    print_vectype_t(a1);
    print_vectype_t(a2);
    print_vectype_t(a3);
    print_vectype_t(a4);
}

void bench_op5_v2(void (*f)(vectype_t &, const vectype_t &, const vectype_t &, const vectype_t &, const Goldilocks::Element[48]))
{
    vectype_t r1, r2, r3, r4, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4;

    const Goldilocks::Element m[48] = {
        0xF21444035865C495,
        0x43921AD645273FCA,
        0xCEE799B27E1D591B,
        0x803427D670647785,
        0x50F4E010CDFDB988,
        0x07D09BF06B814F44,
        0x95F63E9D95D16DC5,
        0xE35B4C3B2D2C9747,
        0xB0C44424817719A1,
        0xBF337BFE194CE1F9,
        0x9DA5789F80E433BD,
        0x25DB2C7514E1D098,
        0x4418DD600F7A8674,
        0xE8A0128436D75EC9,
        0x2B98904D5329B11D,
        0x9C9CF1739732DB2B,
        0xA59C0F80E79533FC,
        0x5753E740275B406D,
        0xB268D66C9C84FD1E,
        0x82375AA17601C78A,
        0xF49FB6AD6682DA33,
        0x4F7CCFE3DD4822C0,
        0x47209122F3BC5DBA,
        0x0134C32CB88338B3,
        0x80DE4F2AB217D233,
        0x5C93E4E4F5BBB1F8,
        0x8A7C8532B6825A31,
        0x199138223483900A,
        0x8D3E5A0B97E211B8,
        0x7A3520B918B4EB7B,
        0x134EF17FFC361854,
        0x6C6B0DE53D119D13,
        0x3CAB68B7918A4B4F,
        0x125735B9DDEDF6CC,
        0xF1AC9E087118C54C,
        0xF020C093C0B8E75B,
        0x4E81A8BB46F98DF2,
        0x37925C785FB36282,
        0x1B34B079A2A54B23,
        0xF62F828EE0ECDD6F,
        0x8F734D4035ACD5F4,
        0x9BAEC668F100FB43,
        0x42D9849AD4647E63,
        0x660546C3B7F60D28,
        0x97A40C51DA303FC3,
        0xBF56548A9481AAA3,
        0x05D180CF6599A290,
        0x6383EF51A90DF3A7};

    struct timeval start, end;

    srand(time(NULL));

#if defined(__USE_SVE__)
    uint64_t r = DELTA + rand();
    a1 = svdup_u64(r);
    a2 = svdup_u64(r);
    a3 = svdup_u64(r);
    a4 = svdup_u64(r);
    b1 = svdup_u64(r);
    b2 = svdup_u64(r);
    b3 = svdup_u64(r);
    b4 = svdup_u64(r);
    c1 = svdup_u64(r);
    c2 = svdup_u64(r);
    c3 = svdup_u64(r);
    c4 = svdup_u64(r);
#elif defined(__USE_AVX__)
    int64_t r = DELTA + rand();
    a1 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a2 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a3 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    a4 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b1 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b2 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b3 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    b4 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    c1 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    c2 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    c3 = _mm256_set_epi64x(r, r, r, r);
    // r = rand();
    c4 = _mm256_set_epi64x(r, r, r, r);
#endif

#ifdef __USE_PAPI__
    papi_init();
#endif
    uint64_t tt = 0;
    gettimeofday(&start, NULL);
    for (int r = 0; r < REPS; r++)
    {
        for (uint64_t i = 0; i < MAX_ITER; i++)
        {
            f(r1, a1, b1, c1, m);
            f(r2, a2, b2, c2, m);
            f(r3, a3, b3, c3, m);
            f(r4, a4, b4, c4, m);
            f(a1, r1, b1, c1, m);
            f(a2, r2, b2, c2, m);
            f(a3, r3, b3, c3, m);
            f(a4, r4, b4, c4, m);
        }
    }
    gettimeofday(&end, NULL);
    tt = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 - start.tv_usec;
#ifdef __USE_PAPI__
    papi_stop();
#endif

    double at = tt / (1000.0f * REPS);
    printf("Elapsed time: %3.4lf ms\n", at);
    printf("Throughput: %3.4lf Mops/s\n\n", 8 * MAX_ITER / at);

    print_vectype_t(a1);
    print_vectype_t(a2);
    print_vectype_t(a3);
    print_vectype_t(a4);
}

int main(int argc, char **argv)
{
    Goldilocks g;

    if (argc < 2)
    {

        printf("Bench add\n");
        f = fopen("add.bin", "wt");
        bench_op3_v1(FUNC1(g, add));
        fclose(f);

        printf("Bench sub\n");
        f = fopen("sub.bin", "wt");
        bench_op3_v1(FUNC1(g, sub));
        fclose(f);

        printf("Bench square_128\n");
        f = fopen("square_128.bin", "wt");
        bench_op3_v2(FUNC2(g, square, 128));
        fclose(f);

        printf("Bench reduce_128_64\n");
        f = fopen("reduce_128_64.bin", "wt");
        bench_op3_v1(FUNC2(g, reduce, 128_64));
        fclose(f);

        printf("Bench mult_128\n");
        f = fopen("mult_128.bin", "wt");
        bench_op4_v1(FUNC2(g, mult, 128));
        fclose(f);

        printf("Bench mult_72\n");
        f = fopen("mult_72.bin", "wt");
        bench_op4_v1(FUNC2(g, mult, 72));
        fclose(f);

        printf("Bench spmv_4x12\n");
        f = fopen("spmv_4x12.bin", "wt");
        bench_op5_v1(FUNC2(g, spmv, 4x12));
        fclose(f);

        printf("Bench spmv_4x12_8\n");
        f = fopen("spmv_4x12_8.bin", "wt");
        bench_op5_v1(FUNC2(g, spmv, 4x12_8));
        fclose(f);

        printf("Bench mmult_4x12\n");
        f = fopen("mmult_4x12.bin", "wt");
        bench_op5_v2(FUNC2(g, mmult, 4x12));
        fclose(f);

        printf("Bench mmult_4x12_8\n");
        f = fopen("mmult_4x12_8.bin", "wt");
        bench_op5_v2(FUNC2(g, mmult, 4x12_8));
        fclose(f);
    }
    else
    {
        int b = atoi(argv[1]);
        switch (b)
        {
        case 0:
            printf("Bench add\n");
            f = fopen("add.bin", "wt");
            bench_op3_v1(FUNC1(g, add));
            fclose(f);
            break;
        case 1:
            printf("Bench sub\n");
            f = fopen("sub.bin", "wt");
            bench_op3_v1(FUNC1(g, sub));
            fclose(f);
            break;
        case 2:
            printf("Bench square_128\n");
            f = fopen("square_128.bin", "wt");
            bench_op3_v2(FUNC2(g, square, 128));
            fclose(f);
            break;
        case 3:
            printf("Bench reduce_128_64\n");
            f = fopen("reduce_128_64.bin", "wt");
            bench_op3_v1(FUNC2(g, reduce, 128_64));
            fclose(f);
            break;
        case 4:
            printf("Bench mult_128\n");
            f = fopen("mult_128.bin", "wt");
            bench_op4_v1(FUNC2(g, mult, 128));
            fclose(f);
            break;
        case 5:
            printf("Bench mult_72\n");
            f = fopen("mult_72.bin", "wt");
            bench_op4_v1(FUNC2(g, mult, 72));
            fclose(f);
            break;
        case 6:
            printf("Bench spmv_4x12\n");
            f = fopen("spmv_4x12.bin", "wt");
            bench_op5_v1(FUNC2(g, spmv, 4x12));
            fclose(f);
            break;
        case 7:
            printf("Bench spmv_4x12_8\n");
            f = fopen("spmv_4x12_8.bin", "wt");
            bench_op5_v1(FUNC2(g, spmv, 4x12_8));
            fclose(f);
            break;
        case 8:
            printf("Bench mmult_4x12\n");
            f = fopen("mmult_4x12.bin", "wt");
            bench_op5_v2(FUNC2(g, mmult, 4x12));
            fclose(f);
            break;
        case 9:
            printf("Bench mmult_4x12_8\n");
            f = fopen("mmult_4x12_8.bin", "wt");
            bench_op5_v2(FUNC2(g, mmult, 4x12_8));
            fclose(f);
            break;

        default:
            printf("Unknown bench id %d!\n", b);
        }
    }

    return 0;
}
