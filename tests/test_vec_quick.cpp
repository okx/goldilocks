#if defined(__USE_SVE__)

#include <sys/prctl.h>
#include "../src/goldilocks_base_field_sve.hpp"
#define VEC sve
typedef svuint64_t vectype_t;

#elif defined(__USE_AVX__)
#if defined(AVX512)

#include "../src/goldilocks_base_field_avx512.hpp"
#define VEC avx512
typedef __m512i vectype_t;

#else

#include "../src/goldilocks_base_field_avx.hpp"
#define VEC avx
typedef __m256i vectype_t;

#endif // AVX512

#endif // __USE

void print_vectype_t(const vectype_t &a)
{
    uint64_t buffer[4];
#ifdef __USE_SVE__
    svst1_u64(svptrue_b64(), buffer, a);
#elif defined(__USE_AVX__)
    _mm256_storeu_si256((__m256i *)buffer, a);
#endif
    for (int i = 0; i < 4; i++)
    {
        printf("0x%lX, ", buffer[i]);
    }
    printf("\n");
}

int is_equal(vectype_t &a, const vectype_t &b)
{
#if defined(__USE_SVE__)
    svbool_t comp = svcmpeq_u64(svptrue_b64(), a, b);
    svbool_t ncomp = svnot_z(svptrue_b64(), comp);
    return !svptest_any(svptrue_b64(), ncomp);
#elif defined(__USE_AVX__)
    uint64_t ru64[4] = {0};
    __m256i r = _mm256_cmpeq_epi64(a, b);
    _mm256_storeu_si256((__m256i *)ru64, r);
    return (ru64[0] == 0xFFFFFFFFFFFFFFFF &&
            ru64[1] == 0xFFFFFFFFFFFFFFFF &&
            ru64[2] == 0xFFFFFFFFFFFFFFFF &&
            ru64[3] == 0xFFFFFFFFFFFFFFFF);
#endif
}

const uint64_t a[4] = {0x2D34400001D1E159, 0x65403422B2810798, 0x32C8E58040D6FC6D, 0xAD9626433BC76360};
const uint64_t b[4] = {0x35821EA14A86CBAA, 0xC989B9E86A0C22D4, 0x136692D5744E3701, 0x2C358311FB7FD2F5};
const uint64_t a0[4] = {0x81CEE3D70A501341, 0xB9F336ACFAD566E8, 0xB3309B7384B412A5, 0x52D9626433BC7636};
const uint64_t a1[4] = {0x9AC10F50A54365D5, 0x62A058ECDF374A63, 0xEDCCD037CDDEA1C5, 0x8D3F3177897FFC89};
const uint64_t a2[4] = {0x6BED18839B7723F1, 0x538142DDD543B214, 0x3377D174424AFC08, 0xF7BC2ED4A9FCD780};
const uint64_t cref[4] = {0x62B65EA14C58AD03, 0x2EC9EE0C1C8D2A6B, 0x462F7855B525336E, 0xD9CBA95537473655};
const uint64_t dref[4] = {0xF7B2215DB74B15B0, 0x9BB67A394874E4C5, 0x1F6252AACC88C56C, 0x8160A3314047906B};
const uint64_t sqhref[4] = {0x7FB692A10A48765, 0x280BB93D7CD12DBA, 0xA13174D7F77FEC6, 0x75B447C7BB4D915F};
const uint64_t sqlref[4] = {0xC6D7CFD46BAF90F1, 0xDD373F80769AA40, 0x8553A1C17F22C669, 0xBE540759E5D36400};
const uint64_t redref[4] = {0x3753FFFA1B80AA51, 0x7C0AC180524AE719, 0x543D8F4200AE5514, 0x67FCE67112224952};
const uint64_t mul128href[4] = {0x972CB311AE68843, 0x4FB5E1D276868AD3, 0x3D94234E1066F32, 0x1DFA178981477B76};
const uint64_t mul128lref[4] = {0x8D93476B51A7381A, 0x9B654D6240FC79E0, 0x9FD0D001BE48676D, 0x4604A51B31F6DAE0};
const uint64_t mul72href[4] = {0xD28E7CF, 0x29F1625F, 0x17128C1D, 0xAA88E40C};
const uint64_t mul72lref[4] = {0xF3FA207251A7381A, 0x2FA193A240FC79E0, 0x668C9F50BE48676D, 0x330CEBBB31F6DAE0};
const uint64_t spmvref[4] = {0xCA6D016470C6B6D7, 0x96A04D1ED64B1D86, 0xFBDC46C8738DC830, 0x7A5C84A3AB928CD2};
const uint64_t spmv8ref[4] = {0xCA6D016470C6B6D7, 0x96A04D1ED64B1D86, 0xFBDC46C8738DC830, 0x7A5C84A3AB928CD2};
const uint64_t mmultref[4] = {0x337AED673F61EC7, 0x76EEAD38017BDE32, 0x9C8237882F9D9B72, 0x94EC23B006B1C58B};
const uint64_t mmult8ref[4] = {0x655B96B89203377E, 0x1FA9FE6C1F5AFC42, 0xD9F8661EACB2C107, 0x9446CDD23A0A85CB};

void test()
{
    Goldilocks g;
    vectype_t ax, bx, cx, cxref, dx, dxref, clx, chx, clxref, chxref;
    const Goldilocks::Element b8[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const Goldilocks::Element m48[48] = {
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

    const Goldilocks::Element m48_8[48] = {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96};

#if defined(__USE_SVE__)
    int vl_in_bytes = prctl(PR_SVE_GET_VL) & PR_SVE_VL_LEN_MASK;
    if (vl_in_bytes < 32) {
        printf("FATAL: SVE vector length should be at least 256 bits (it is currently %d bits)!\n", vl_in_bytes * 8);
        return;
    }

    ax = svld1_u64(svptrue_b64(), (uint64_t *)a);
    bx = svld1_u64(svptrue_b64(), (uint64_t *)b);
    cxref = svld1_u64(svptrue_b64(), (uint64_t *)cref);
    dxref = svld1_u64(svptrue_b64(), (uint64_t *)dref);
    clxref = svld1_u64(svptrue_b64(), (uint64_t *)sqlref);
    chxref = svld1_u64(svptrue_b64(), (uint64_t *)sqhref);

    g.add_sve(cx, ax, bx);
    assert(is_equal(cx, cxref));

    g.sub_sve(dx, ax, bx);
    assert(is_equal(dx, dxref));

    g.square_sve_128(chx, clx, ax);
    assert(is_equal(chx, chxref));
    assert(is_equal(clx, clxref));

    g.reduce_sve_128_64(cx, ax, bx);
    cxref = svld1_u64(svptrue_b64(), (uint64_t *)redref);
    assert(is_equal(cx, cxref));

    g.mult_sve_128(chx, clx, ax, bx);
    chxref = svld1_u64(svptrue_b64(), (uint64_t *)mul128href);
    clxref = svld1_u64(svptrue_b64(), (uint64_t *)mul128lref);
    assert(is_equal(chx, chxref));
    assert(is_equal(clx, clxref));

    g.mult_sve_72(chx, clx, ax, bx);
    chxref = svld1_u64(svptrue_b64(), (uint64_t *)mul72href);
    clxref = svld1_u64(svptrue_b64(), (uint64_t *)mul72lref);
    assert(is_equal(chx, chxref));
    assert(is_equal(clx, clxref));

    ax = svld1_u64(svptrue_b64(), (uint64_t *)a0);
    bx = svld1_u64(svptrue_b64(), (uint64_t *)a1);
    cx = svld1_u64(svptrue_b64(), (uint64_t *)a2);

    g.spmv_sve_4x12(dx, ax, bx, cx, b8);
    dxref = svld1_u64(svptrue_b64(), (uint64_t *)spmvref);
    assert(is_equal(dx, dxref));

    g.spmv_sve_4x12_8(dx, ax, bx, cx, b8);
    dxref = svld1_u64(svptrue_b64(), (uint64_t *)spmv8ref);
    assert(is_equal(dx, dxref));

    g.mmult_sve_4x12(dx, ax, bx, cx, m48);
    dxref = svld1_u64(svptrue_b64(), (uint64_t *)mmultref);
    assert(is_equal(dx, dxref));

    g.mmult_sve_4x12_8(dx, ax, bx, cx, m48_8);
    dxref = svld1_u64(svptrue_b64(), (uint64_t *)mmult8ref);
    assert(is_equal(dx, dxref));
#else
    ax = _mm256_set_epi64x(a[3], a[2], a[1], a[0]);
    bx = _mm256_set_epi64x(b[3], b[2], b[1], b[0]);
    cxref = _mm256_set_epi64x(cref[3], cref[2], cref[1], cref[0]);
    dxref = _mm256_set_epi64x(dref[3], dref[2], dref[1], dref[0]);
    clxref = _mm256_set_epi64x(sqlref[3], sqlref[2], sqlref[1], sqlref[0]);
    chxref = _mm256_set_epi64x(sqhref[3], sqhref[2], sqhref[1], sqhref[0]);

    g.add_avx(cx, ax, bx);
    assert(is_equal(cx, cxref));

    g.sub_avx(dx, ax, bx);
    assert(is_equal(dx, dxref));

    g.square_avx_128(chx, clx, ax);
    assert(is_equal(chx, chxref));
    assert(is_equal(clx, clxref));

    g.reduce_avx_128_64(cx, ax, bx);
    cxref = _mm256_set_epi64x(redref[3], redref[2], redref[1], redref[0]);
    assert(is_equal(cx, cxref));

    g.mult_avx_128(chx, clx, ax, bx);
    chxref = _mm256_set_epi64x(mul128href[3], mul128href[2], mul128href[1], mul128href[0]);
    clxref = _mm256_set_epi64x(mul128lref[3], mul128lref[2], mul128lref[1], mul128lref[0]);
    assert(is_equal(chx, chxref));
    assert(is_equal(clx, clxref));

    g.mult_avx_72(chx, clx, ax, bx);
    chxref = _mm256_set_epi64x(mul72href[3], mul72href[2], mul72href[1], mul72href[0]);
    clxref = _mm256_set_epi64x(mul72lref[3], mul72lref[2], mul72lref[1], mul72lref[0]);
    assert(is_equal(chx, chxref));
    assert(is_equal(clx, clxref));

    ax = _mm256_set_epi64x(a0[3], a0[2], a0[1], a0[0]);
    bx = _mm256_set_epi64x(a1[3], a1[2], a1[1], a1[0]);
    cx = _mm256_set_epi64x(a2[3], a2[2], a2[1], a2[0]);

    g.spmv_avx_4x12(dx, ax, bx, cx, b8);
    dxref = _mm256_set_epi64x(spmvref[3], spmvref[2], spmvref[1], spmvref[0]);
    assert(is_equal(dx, dxref));

    g.spmv_avx_4x12_8(dx, ax, bx, cx, b8);
    dxref = _mm256_set_epi64x(spmv8ref[3], spmv8ref[2], spmv8ref[1], spmv8ref[0]);
    assert(is_equal(dx, dxref));

    g.mmult_avx_4x12(dx, ax, bx, cx, m48);
    dxref = _mm256_set_epi64x(mmultref[3], mmultref[2], mmultref[1], mmultref[0]);
    assert(is_equal(dx, dxref));

    g.mmult_avx_4x12_8(dx, ax, bx, cx, m48_8);
    dxref = _mm256_set_epi64x(mmult8ref[3], mmult8ref[2], mmult8ref[1], mmult8ref[0]);
    assert(is_equal(dx, dxref));
#endif

    printf("All tests done.\n");
}

int main()
{
    test();

    return 0;
}