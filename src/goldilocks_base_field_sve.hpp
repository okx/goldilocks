#ifndef GOLDILOCKS_SVE
#define GOLDILOCKS_SVE

#include "goldilocks_base_field.hpp"
#include <string.h>

#define __USE_SVE_ASM__

// NOTATION:
// _c value is in canonical form
// _s value shifted (a_s = a + (1<<63) = a XOR (1<<63)
// _n negative P_n = -P
// _l low part of a variable: uint64 [31:0] or uint128 [63:0]
// _h high part of a variable: uint64 [63:32] or uint128 [127:64]
// _a alingned pointer
// _8 variable can be expressed in 8 bits (<256)

// OBSERVATIONS:
// We do not work with shifted values. We work with packed uint64_t.

#define GP svdup_u64(GOLDILOCKS_PRIME)
#define P_n svdup_u64(GOLDILOCKS_PRIME_NEG)
#define SQMASK svdup_u64(0x1FFFFFFFF)
#define LMASK svdup_u64(0xFFFFFFFF)

inline void Goldilocks::set_sve(svuint64_t &a, const Goldilocks::Element &a0, const Goldilocks::Element &a1, const Goldilocks::Element &a2, const Goldilocks::Element &a3)
{
    uint64_t base[4] = {a0.fe, a1.fe, a2.fe, a3.fe};
    a = svld1_u64(svptrue_b64(), (uint64_t *)base);
}

inline void Goldilocks::load_sve(svuint64_t &a, const Goldilocks::Element *a4)
{
    a = svld1_u64(svptrue_b64(), (uint64_t *)a4);
}

// We assume a4_a aligned on a 32-byte boundary
inline void Goldilocks::load_sve_a(svuint64_t &a, const Goldilocks::Element *a4_a)
{
    a = svld1_u64(svptrue_b64(), (uint64_t *)a4_a);
}

inline void Goldilocks::store_sve(Goldilocks::Element *a4, const svuint64_t &a)
{
    svst1_u64(svptrue_b64(), (uint64_t *)a4, a);
}

// We assume a4_a aligned on a 32-byte boundary
inline void Goldilocks::store_sve_a(Goldilocks::Element *a4_a, const svuint64_t &a)
{
    svst1_u64(svptrue_b64(), (uint64_t *)a4_a, a);
}

inline void Goldilocks::add_sve(svuint64_t &c, const svuint64_t &a, const svuint64_t &b)
{
#ifndef __USE_SVE_ASM__
    svuint64_t c1 = svadd_u64_z(svptrue_b64(), a, b);
    svuint64_t d = svsub_u64_z(svptrue_b64(), GP, a);
    svuint64_t c2 = svadd_u64_z(svptrue_b64(), c1, P_n);
    svbool_t mask_ = svcmpgt_u64(svptrue_b64(), b, d);
    c = svsel_u64(mask_, c2, c1);
#else
    asm inline("ptrue   p7.b, all\n"
               "ld1d    z31.d, p7/z, %1\n"
               "ld1d    z29.d, p7/z, %2\n"
               "mov     z28.d, #4294967295\n"
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z31.d\n"
               "add     z31.d, z29.d, z31.d\n"
               "cmphi   p6.d, p7/z, z29.d, z30.d\n"
               "add     z31.d, p6/m, z31.d, z28.d\n"
               "st1d    z31.d, p7, %0\n"
               : "=m"(c)
               : "m"(a), "m"(b));
#endif // __USE_SVE_ASM__
}

inline void Goldilocks::sub_sve(svuint64_t &c, const svuint64_t &a, const svuint64_t &b)
{
#ifndef __USE_SVE_ASM__
    svuint64_t c1 = svsub_u64_z(svptrue_b64(), a, b);
    svbool_t mask_ = svcmpge_u64(svptrue_b64(), a, b);
    svuint64_t c2 = svadd_u64_z(svptrue_b64(), c1, GP);
    c = svsel_u64(mask_, c1, c2);
#else
    asm inline("ptrue   p7.b, all\n"
               "ld1d    z31.d, p7/z, %1\n"
               "ld1d    z29.d, p7/z, %2\n"
               "mov     z30.d, #-4294967295\n"
               "sub     z28.d, z31.d, z29.d\n"
               "cmphs   p6.d, p7/z, z31.d, z29.d\n"
               "add     z30.d, z30.d, z28.d\n"
               "sel     z30.d, p6, z28.d, z30.d\n"
               "st1d    z30.d, p7, %0\n"
               : "=m"(c)
               : "m"(a), "m"(b));
#endif // __USE_SVE_ASM__
}

inline void Goldilocks::mult_sve(svuint64_t &c, const svuint64_t &a, const svuint64_t &b)
{
    svuint64_t c_h, c_l;
    mult_sve_128(c_h, c_l, a, b);
    reduce_sve_128_64(c, c_h, c_l);
}

inline void Goldilocks::mult_sve_reg(svuint64_t &c, const svuint64_t &a, const Goldilocks::Element *b)
{
#ifdef __USE_SVE_ASM__
    asm inline("ptrue   p7.d, all\n"
               "ld1d    z31.d, p7/z, %1\n"
               "ld1d    z30.d, p7/z, [%2]\n"
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "mov     z23.d, z31.d\n"
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z25.d, z24.d, #32\n"         // r1_h
               "add     z25.d, p7/m, z25.d, z29.d\n" // c_h
               "mov     z31.d, z25.d\n"
               "lsl     z25.d, z24.d, #32\n" // r1_l
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z29.s, p15, z25.s, z23.s\n"  // c_l
               "lsr     z30.d, z31.d, #32\n"         // c_hh
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z27.d, p6, z28.d, z27.d\n"   // c1
               "mov     z29.d, #0xFFFFFFFF\n"        // P_n
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z27.d\n" // GP - c1 (GP-a)
               "mul     z31.d, p7/m, z31.d, z29.d\n" // c2 (c1 + c_hl * P_n)
               "add     z27.d, p7/m, z27.d, z31.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z31.d, z30.d\n"
               "add     z27.d, p6/m, z27.d, z29.d\n" // + P_n
               "st1d    z27.d, p7, %0\n"
               : "=m"(c)
               : "m"(a), "r"((uint64_t *)b));
#else
    assert(0);
#endif
}

// We assume coeficients of b_8 can be expressed with 8 bits (<256)
inline void Goldilocks::mult_sve_8(svuint64_t &c, const svuint64_t &a, const svuint64_t &b_8)
{
    svuint64_t c_h, c_l;
    mult_sve_72(c_h, c_l, a, b_8);
    reduce_sve_96_64(c, c_h, c_l);
}

// The 128 bits of the result are stored in c_h[64:0]| c_l[64:0]
inline void Goldilocks::mult_sve_128(svuint64_t &c_h, svuint64_t &c_l, const svuint64_t &a, const svuint64_t &b)
{
#ifndef __USE_SVE_ASM__
    // Split into 32 bits
    svuint64_t a_h = svlsr_n_u64_z(svptrue_b64(), a, 32);
    svuint64_t b_h = svlsr_n_u64_z(svptrue_b64(), b, 32);
    svuint64_t a_l = svand_u64_z(svptrue_b64(), a, LMASK);
    svuint64_t b_l = svand_u64_z(svptrue_b64(), b, LMASK);

    // c = (a_h + a_l) * (b_h + b_l)
    svuint64_t c_hh = svmul_u64_z(svptrue_b64(), a_h, b_h);
    svuint64_t c_hl = svmul_u64_z(svptrue_b64(), a_h, b_l);
    svuint64_t c_lh = svmul_u64_z(svptrue_b64(), a_l, b_h);
    svuint64_t c_ll = svmul_u64_z(svptrue_b64(), a_l, b_l);

    svuint64_t c_ll_h = svlsr_n_u64_z(svptrue_b64(), c_ll, 32);
    svuint64_t r0 = svadd_u64_z(svptrue_b64(), c_hl, c_ll_h);
    svuint64_t r0_l = svand_u64_z(svptrue_b64(), r0, P_n);
    svuint64_t r1 = svadd_u64_z(svptrue_b64(), c_lh, r0_l);
    svuint64_t r1_l = svlsl_n_u64_z(svptrue_b64(), r1, 32);
    const svbool_t sel = svdupq_n_b32(0, 1, 0, 1);
    c_l = svreinterpret_u64(svsel_u32(sel, svreinterpret_u32(r1_l), svreinterpret_u32(c_ll)));
    svuint64_t r0_h = svlsr_n_u64_z(svptrue_b64(), r0, 32);
    svuint64_t r2 = svadd_u64_z(svptrue_b64(), c_hh, r0_h);
    svuint64_t r1_h = svlsr_n_u64_z(svptrue_b64(), r1, 32);
    c_h = svadd_u64_z(svptrue_b64(), r2, r1_h);
#else
    // z31 <- a (a_l), z30 <- b (b_l)
    // z29 <- a_h, z28 <- b_h
    asm inline("ptrue    p7.d, all\n"
               "ld1d    z31.d, p7/z, %2\n"
               "ld1d    z30.d, p7/z, %3\n"
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z25.d, z24.d, #32\n"         // r1_h
               "add     z25.d, p7/m, z25.d, z29.d\n" // c_h
               "st1d    z25.d, p7, %0\n"
               "lsl     z25.d, z24.d, #32\n" // r1_l
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z25.s, p15, z25.s, z31.s\n"  // c_l
               "st1d    z25.d, p7, %1\n"
               : "=m"(c_h), "=m"(c_l)
               : "m"(a), "m"(b));
#endif // __USE_SVE_ASM__
}

// The 72 bits the result are stored in c_h[32:0] | c_l[64:0]
inline void Goldilocks::mult_sve_72(svuint64_t &c_h, svuint64_t &c_l, const svuint64_t &a, const svuint64_t &b)
{
#ifndef __USE_SVE_ASM__
    svuint64_t a_h = svlsr_n_u64_z(svptrue_b64(), a, 32);
    svuint64_t a_l = svand_u64_z(svptrue_b64(), a, LMASK);
    svuint64_t b_l = svand_u64_z(svptrue_b64(), b, LMASK);
    svuint64_t c_hl = svmul_u64_z(svptrue_b64(), a_h, b_l);
    svuint64_t c_ll = svmul_u64_z(svptrue_b64(), a_l, b_l);
    svuint64_t c_ll_h = svlsr_n_u64_z(svptrue_b64(), c_ll, 32);
    svuint64_t r0 = svadd_u64_z(svptrue_b64(), c_hl, c_ll_h);
    svuint64_t r0_l = svlsl_n_u64_z(svptrue_b64(), r0, 32);
    const svbool_t sel = svdupq_n_b32(0, 1, 0, 1);
    c_l = svreinterpret_u64(svsel_u32(sel, svreinterpret_u32(r0_l), svreinterpret_u32(c_ll)));
    c_h = svlsr_n_u64_z(svptrue_b64(), r0, 32);
#else
    asm inline("ptrue   p7.d, all\n"
               "ld1d    z31.d, p7/z, %2\n"
               "ld1d    z30.d, p7/z, %3\n"
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z29.d, z29.d, #32\n"         // c_h
               "st1d    z29.d, p7, %0\n"
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z30.s, p15, z30.s, z31.s\n"
               "st1d    z30.d, p7, %1\n"
               : "=m"(c_h), "=m"(c_l)
               : "m"(a), "m"(b));
#endif // __USE_SVE_ASM__
}

inline void Goldilocks::mult_sve_72_reg(svuint64_t &c_h, svuint64_t &c_l, const svuint64_t &a, const Goldilocks::Element *b)
{
#ifdef __USE_SVE_ASM__
    asm inline("ptrue   p7.d, all\n"
               "ld1d    z31.d, p7/z, %2\n"
               "ld1d    z30.d, p7/z, [%3]\n"
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z29.d, z29.d, #32\n"         // c_h
               "st1d    z29.d, p7, %0\n"
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z30.s, p15, z30.s, z31.s\n"
               "st1d    z30.d, p7, %1\n"
               : "=m"(c_h), "=m"(c_l)
               : "m"(a), "r"((uint64_t *)b));
#else
    assert(0);
#endif
}

// notes:
// 2^64 = P+P_n => [2^64]=[P_n]
// P = 2^64-2^32+1
// P_n = 2^32-1
// 2^32*P_n = 2^32*(2^32-1) = 2^64-2^32 = P-1
// process:
// c % P = [c] = [c_h*2^64+c_l] = [c_h*P_n+c_l] = [c_hh*2^32*P_n+c_hl*P_n+c_l] =
//             = [c_hh(P-1) +c_hl*P_n+c_l] = [c_l-c_hh+c_hl*P_n]
inline void Goldilocks::reduce_sve_128_64(svuint64_t &c, const svuint64_t &c_h, const svuint64_t &c_l)
{
#ifndef __USE_SVE_ASM__
    svuint64_t c_hh = svlsr_n_u64_z(svptrue_b64(), c_h, 32);
    svuint64_t c_hl = svand_u64_z(svptrue_b64(), c_h, LMASK);
    svuint64_t c1;
    sub_sve(c1, c_l, c_hh);
    svuint64_t c2 = svmul_u64_z(svptrue_b64(), c_hl, P_n); // c_hl*P_n (only 32bits of c_h useds)
    add_sve(c, c1, c2);
#else
    asm inline("ptrue   p7.b, all\n"
               "ld1d    z31.d, p7/z, %1\n"           // c_h
               "lsr     z30.d, z31.d, #32\n"         // c_hh
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // c_hl
               "ld1d    z29.d, p7/z, %2\n"           // c_l
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z27.d, p6, z28.d, z27.d\n"   // c1
               "mov     z29.d, #0xFFFFFFFF\n"        // P_n
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z27.d\n" // GP - c1 (GP-a)
               "mul     z31.d, p7/m, z31.d, z29.d\n" // c2 (c1 + c_hl * P_n)
               "add     z27.d, p7/m, z27.d, z31.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z31.d, z30.d\n"
               "add     z27.d, p6/m, z27.d, z29.d\n" // + P_n
               "st1d    z27.d, p7, %0\n"
               : "=m"(c)
               : "m"(c_h), "m"(c_l));
#endif // __USE_SVE_ASM__
}

// notes:
// P = 2^64-2^32+1
// P_n = 2^32-1
// 2^32*P_n = 2^32*(2^32-1) = 2^64-2^32 = P-1
// 2^64 = P+P_n => [2^64]=[P_n]
// c_hh = 0 in this case
// process:
// c % P = [c] = [c_h*1^64+c_l] = [c_h*P_n+c_l] = [c_hh*2^32*P_n+c_hl*P_n+c_l] =
//             = [c_hl*P_n+c_l] = [c_l+c_hl*P_n]
inline void Goldilocks::reduce_sve_96_64(svuint64_t &c, const svuint64_t &c_h, const svuint64_t &c_l)
{
    svuint64_t c1 = svmul_u64_z(svptrue_b64(), c_h, P_n); // c_hl*P_n (only 32bits of c_h useds)
    add_sve(c, c_l, c1);                                  // c1 = c_hl*P_n <= (2^32-1)*(2^32-1) <= 2^64 -2^33+1 < P
}

inline void Goldilocks::square_sve(svuint64_t &c, svuint64_t &a)
{
    svuint64_t c_h, c_l;
    square_sve_128(c_h, c_l, a);
    reduce_sve_128_64(c, c_h, c_l);
}

inline void Goldilocks::square_sve_128(svuint64_t &c_h, svuint64_t &c_l, const svuint64_t &a)
{
#ifndef __USE_SVE_ASM__
    svuint64_t a_h = svlsr_n_u64_z(svptrue_b64(), a, 32);
    svuint64_t a_l = svand_u64_z(svptrue_b64(), a, LMASK);
    svuint64_t c_hh = svmul_u64_z(svptrue_b64(), a_h, a_h);
    svuint64_t c_lh = svmul_u64_z(svptrue_b64(), a_l, a_h);
    svuint64_t c_ll = svmul_u64_z(svptrue_b64(), a_l, a_l);
    svuint64_t c_ll_h = svlsr_n_u64_z(svptrue_b64(), c_ll, 33); // yes 33, low part of 2*c_lh is [31:0]
    svuint64_t r0 = svadd_u64_z(svptrue_b64(), c_lh, c_ll_h);
    svuint64_t r0_l = svlsl_n_u64_z(svptrue_b64(), r0, 33);
    svuint64_t c_ll_l = svand_u64_z(svptrue_b64(), c_ll, SQMASK);
    c_l = svadd_u64_z(svptrue_b64(), r0_l, c_ll_l);
    svuint64_t r0_h = svlsr_n_u64_z(svptrue_b64(), r0, 31);
    c_h = svadd_u64_z(svptrue_b64(), c_hh, r0_h);
#else
    asm inline("ptrue   p7.b, all\n"
               "ld1d    z31.d, p7/z, %2\n"           // a
               "lsr     z30.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "mov     z29.d, z31.d\n"
               "mul     z29.d, p7/m, z29.d, z31.d\n"  // c_ll = a_l * a_l
               "lsr     z28.d, z29.d, #33\n"          // c_ll_h
               "and     z29.d, z29.d, #0x1FFFFFFFF\n" // c_ll_l
               "mad     z31.d, p7/m, z30.d, z28.d\n"  // r0
               "lsl     z28.d, z31.d, #33\n"          // r0_l
               "add     z28.d, p7/m, z28.d, z29.d\n"  // c_l
               "lsr     z31.d, z31.d, #31\n"          // r0_h
               "mad     z30.d, p7/m, z30.d, z31.d\n"  // c_h
               "st1d    z28.d, p7, %1\n"
               "st1d    z30.d, p7, %0\n"
               : "=m"(c_h), "=m"(c_l) // %0, %1
               : "m"(a));             // %2
#endif // __USE_SVE_ASM__
}

inline Goldilocks::Element Goldilocks::dot_sve(const svuint64_t &a0, const svuint64_t &a1, const svuint64_t &a2, const Element b[12])
{
    svuint64_t c_;
    spmv_sve_4x12(c_, a0, a1, a2, b);
    Goldilocks::Element c[4];
    store_sve(c, c_);
    return (c[0] + c[1]) + (c[2] + c[3]);
}

// Sparse matrix-vector product (4x12 sparce matrix formed of three diagonal blocks of size 4x4)
// c[i]=Sum_j(aj[i]*b[j*4+i]) 0<=i<4 0<=j<3
inline void Goldilocks::spmv_sve_4x12(svuint64_t &c, const svuint64_t &a0, const svuint64_t &a1, const svuint64_t &a2, const Goldilocks::Element b[12])
{
    uint64_t *bb = (uint64_t *)b;

#ifdef __SVE_512__
    uint64_t b24[16];
    bb = b24;
    memcpy(bb, b, 96);
    uint64_t idx[8] = {8, 8, 8, 8, 0, 1, 2, 3};
#endif

#ifndef __USE_SVE_ASM__
    svuint64_t b0, b1, b2;
    svuint64_t c0, c1, c2;
    load_sve(b0, (Goldilocks::Element *)&(bb[0]));
    load_sve(b1, (Goldilocks::Element *)&(bb[4]));
    load_sve(b2, (Goldilocks::Element *)&(bb[8]));
#ifdef __SVE_512__
    svuint64_t vidx = svld1_u64(svptrue_b64(), idx);
    b0 = svtbx_u64(b0, b0, vidx);
    b1 = svtbx_u64(b1, b1, vidx);
    b2 = svtbx_u64(b2, b2, vidx);
#endif
    mult_sve(c0, a0, b0);
    mult_sve(c1, a1, b1);
    mult_sve(c2, a2, b2);
    svuint64_t c_;
    add_sve(c_, c0, c1);
    add_sve(c, c_, c2);
#else
    asm inline("ptrue   p7.d, all\n"
               "mov     z22.d, #0xFFFFFFFF\n" // P_n
                                              // --- mul 1
               "ld1d    z31.d, p7/z, %1\n"    // a0 (mul_1)
               "ld1d    z30.d, p7/z, [%4]\n"  // b[0]
#ifdef __SVE_512__
               "ld1d    z10.d, p7/z, [%7]\n"
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z11.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z11.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z11.d, p7/m, z11.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n" // c0 (-> z11)
               // --- mul 2
               "ld1d    z31.d, p7/z, %2\n"   // a1 (mul_2)
               "ld1d    z30.d, p7/z, [%5]\n" // b[4]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z12.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z12.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z12.d, p7/m, z12.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // c0 (c1 -> z12)
               // --- mul 3
               "ld1d    z31.d, p7/z, %3\n"   // a2 (mul_3)
               "ld1d    z30.d, p7/z, [%6]\n" // b[8]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z13.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z13.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z13.d, p7/m, z13.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z13.d, p6/m, z13.d, z22.d\n" // c0 (c2 -> z13)
               // --- add 1 (c0 + c1 = z11 + z12)
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z11.d\n"
               "add     z11.d, z12.d, z11.d\n"
               "cmphi   p6.d, p7/z, z12.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n"
               "mov     z30.d, #-4294967295\n" // add_2 ( + c2 = z11 + z13)
               "sub     z30.d, z30.d, z11.d\n"
               "add     z11.d, z13.d, z11.d\n"
               "cmphi   p6.d, p7/z, z13.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n"
               "st1d    z11.d, p7, %0\n"
               : "=m"(c)
               : "m"(a0), "m"(a1), "m"(a2), "r"((uint64_t *)&bb[0]), "r"((uint64_t *)&bb[4]), "r"((uint64_t *)&bb[8])
#ifdef __SVE_512__
                                                                                                  ,
                 "r"(idx)
#endif
    );
#endif
}

// Sparse matrix-vector product (4x12 sparce matrix formed of four diagonal blocs 4x5 stored in a0...a3)
// c[i]=Sum_j(aj[i]*b[j*4+i]) 0<=i<4 0<=j<3
// We assume b_a aligned on a 32-byte boundary
// We assume coeficients of b_8 can be expressed with 8 bits (<256)
inline void Goldilocks::spmv_sve_4x12_8(svuint64_t &c, const svuint64_t &a0, const svuint64_t &a1, const svuint64_t &a2, const Goldilocks::Element b_8[12])
{
    uint64_t *bb = (uint64_t *)b_8;

#ifdef __SVE_512__
    uint64_t b24[16];
    bb = b24;
    memcpy(bb, b_8, 96);
    uint64_t idx[8] = {8, 8, 8, 8, 0, 1, 2, 3};
#endif

#ifndef __USE_SVE_ASM__
    svuint64_t b0, b1, b2;
    svuint64_t c0_h, c1_h, c2_h;
    svuint64_t c0_l, c1_l, c2_l;
    svuint64_t c_h, c_l, aux_h, aux_l;

    load_sve(b0, (Goldilocks::Element *)&(bb[0]));
    load_sve(b1, (Goldilocks::Element *)&(bb[4]));
    load_sve(b2, (Goldilocks::Element *)&(bb[8]));
#ifdef __SVE_512__
    svuint64_t vidx = svld1_u64(svptrue_b64(), idx);
    b0 = svtbx_u64(b0, b0, vidx);
    b1 = svtbx_u64(b1, b1, vidx);
    b2 = svtbx_u64(b2, b2, vidx);
#endif

    mult_sve_72(c0_h, c0_l, a0, b0);
    mult_sve_72(c1_h, c1_l, a1, b1);
    mult_sve_72(c2_h, c2_l, a2, b2);

    add_sve(aux_l, c0_l, c1_l);
    add_sve(c_l, aux_l, c2_l);

    aux_h = svadd_u64_z(svptrue_b64(), c0_h, c1_h); // do with epi32?
    c_h = svadd_u64_z(svptrue_b64(), aux_h, c2_h);

    reduce_sve_96_64(c, c_h, c_l);
#else
    asm inline("ptrue   p7.d, all\n"
               "ld1d    z31.d, p7/z, %1\n" // a0 (mul_1)
               "ld1d    z30.d, p7/z, [%4]\n"
#ifdef __SVE_512__
               "ld1d    z10.d, p7/z, [%7]\n"
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z11.d, z29.d, #32\n"         // c_h (c0_h -> z11)
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z12.s, p15, z30.s, z31.s\n"  // c_l (c0_l -> z12)
               "ld1d    z31.d, p7/z, %2\n"           // a1  (mul_2)
               "ld1d    z30.d, p7/z, [%5]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z13.d, z29.d, #32\n"         // c_h (c1_h -> z13)
               "sel     z14.s, p15, z30.s, z31.s\n"  // c_l (c1_l -> z14)
               "ld1d    z31.d, p7/z, %3\n"           // a2  (mul_3)
               "ld1d    z30.d, p7/z, [%6]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z15.d, z29.d, #32\n"         // c_h (c2_h -> z15)
               "sel     z16.s, p15, z30.s, z31.s\n"  // c_l (c2_l -> z16)
               "mov     z28.d, #4294967295\n"        // add_sve (z12 + z14)
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z14.d, z12.d\n"
               "cmphi   p6.d, p7/z, z14.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z28.d\n" // aux_l -> z12
               "mov     z30.d, #-4294967295\n"       // add_sve (z12 + z16)
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z16.d, z12.d\n"
               "cmphi   p6.d, p7/z, z16.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z28.d\n" // c_l -> z12
               "add     z11.d, p7/m, z11.d, z13.d\n"
               "add     z11.d, p7/m, z11.d, z15.d\n" // c_h -> z11
               "mul     z11.d, p7/m, z11.d, z28.d\n" // c1
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z11.d, z12.d\n"
               "cmphi   p6.d, p7/z, z11.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z28.d\n"
               "st1d    z12.d, p7, %0\n"
               : "=m"(c)
               : "m"(a0), "m"(a1), "m"(a2), "r"((uint64_t *)(&bb[0])), "r"((uint64_t *)(&bb[4])), "r"((uint64_t *)(&bb[8]))
#ifdef __SVE_512__
                                                                                                      ,
                 "r"(idx)
#endif
    );
#endif
}

inline void Goldilocks::transpose(svuint64_t &r0, svuint64_t &r1, svuint64_t &r2, svuint64_t &r3,
                                  svuint64_t &c0, svuint64_t &c1, svuint64_t &c2, svuint64_t &c3)
{
#ifdef __SVE_512__
    uint64_t idx1[8] = {8, 8, 0, 1, 8, 8, 4, 5};
    uint64_t idx2[8] = {2, 3, 8, 8, 6, 7, 8, 8};
    uint64_t idx3[8] = {8, 0, 8, 2, 8, 4, 8, 6};
    uint64_t idx4[8] = {1, 8, 3, 8, 5, 8, 7, 8};
    svuint64_t vidx1 = svld1_u64(svptrue_b64(), idx1);
    svuint64_t vidx2 = svld1_u64(svptrue_b64(), idx2);
    svuint64_t vidx3 = svld1_u64(svptrue_b64(), idx3);
    svuint64_t vidx4 = svld1_u64(svptrue_b64(), idx4);
    svuint64_t t0 = svtbx_u64(r0, r2, vidx1);
    svuint64_t t1 = svtbx_u64(r1, r3, vidx1);
    svuint64_t t2 = svtbx_u64(r2, r0, vidx2);
    svuint64_t t3 = svtbx_u64(r3, r1, vidx2);
    c0 = svtbx_u64(t0, t1, vidx3);
    c1 = svtbx_u64(t1, t0, vidx4);
    c2 = svtbx_u64(t2, t3, vidx3);
    c3 = svtbx_u64(t3, t2, vidx4);
#else
    svuint64_t t0 = svzip1_u64(r0, r2);
    svuint64_t t1 = svzip1_u64(r1, r3);
    svuint64_t t2 = svzip2_u64(r0, r2);
    svuint64_t t3 = svzip2_u64(r1, r3);
    c0 = svzip1_u64(t0, t1);
    c1 = svzip2_u64(t0, t1);
    c2 = svzip1_u64(t2, t3);
    c3 = svzip2_u64(t2, t3);
#endif
}

// Dense matrix-vector product
inline void Goldilocks::mmult_sve_4x12(svuint64_t &b, const svuint64_t &a0, const svuint64_t &a1, const svuint64_t &a2, const Goldilocks::Element M[48])
{

#ifndef __USE_SVE_ASM__
    // Generate matrix 4x4
    svuint64_t r0, r1, r2, r3;
    Goldilocks::spmv_sve_4x12(r0, a0, a1, a2, &(M[0]));
    Goldilocks::spmv_sve_4x12(r1, a0, a1, a2, &(M[12]));
    Goldilocks::spmv_sve_4x12(r2, a0, a1, a2, &(M[24]));
    Goldilocks::spmv_sve_4x12(r3, a0, a1, a2, &(M[36]));

    // Transpose: transform de 4x4 matrix stored in rows r0...r3 to the columns c0...c3
    svuint64_t c0, c1, c2, c3;
    transpose(r0, r1, r2, r3, c0, c1, c2, c3);

    // Add columns to obtain result
    svuint64_t sum0, sum1;
    add_sve(sum0, c0, c1);
    add_sve(sum1, c2, c3);
    add_sve(b, sum0, sum1);
#else
    uint64_t *mm = (uint64_t *)M;
#ifdef __SVE_512__
    uint64_t idx1[8] = {8, 8, 0, 1, 8, 8, 4, 5};
    uint64_t idx2[8] = {2, 3, 8, 8, 6, 7, 8, 8};
    uint64_t idx3[8] = {8, 0, 8, 2, 8, 4, 8, 6};
    uint64_t idx4[8] = {1, 8, 3, 8, 5, 8, 7, 8};
    uint64_t m52[52];
    mm = m52;
    memcpy(mm, M, 384);
    uint64_t idx[8] = {8, 8, 8, 8, 0, 1, 2, 3};
#endif
    asm inline("ptrue   p7.d, all\n"
               "mov     z22.d, #0xFFFFFFFF\n" // P_n
               // spmv_sve_4x12(r0, a0, a1, a2, &(M[0]));
               // --- mul 1
               "ld1d    z31.d, p7/z, %1\n"   // a0 (mul_1)
               "ld1d    z30.d, p7/z, [%4]\n" // b[0]
#ifdef __SVE_512__
               "ld1d    z10.d, p7/z, [%20]\n"
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z11.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z11.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z11.d, p7/m, z11.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n" // c0 (-> z11)
               // --- mul 2
               "ld1d    z31.d, p7/z, %2\n"   // a1 (mul_2)
               "ld1d    z30.d, p7/z, [%5]\n" // b[4]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z12.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z12.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z12.d, p7/m, z12.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // c0 (c1 -> z12)
               // --- mul 3
               "ld1d    z31.d, p7/z, %3\n"   // a2 (mul_3)
               "ld1d    z30.d, p7/z, [%6]\n" // b[8]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z13.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z13.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z13.d, p7/m, z13.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z13.d, p6/m, z13.d, z22.d\n" // c0 (c2 -> z13)
               // --- add 1 (c0 + c1 = z11 + z12)
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z11.d\n"
               "add     z11.d, z12.d, z11.d\n"
               "cmphi   p6.d, p7/z, z12.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n"
               "mov     z30.d, #-4294967295\n" // add_2 ( + c2 = z11 + z13)
               "sub     z30.d, z30.d, z11.d\n"
               "add     z11.d, z13.d, z11.d\n"
               "cmphi   p6.d, p7/z, z13.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n"
               "mov     z15.d, z11.d\n" // r0
               // *** spmv_sve_4x12(r1, a0, a1, a2, &(M[12]));
               "ld1d    z31.d, p7/z, %1\n"   // a0 (mul_1)
               "ld1d    z30.d, p7/z, [%7]\n" // b[0]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z11.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z11.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z11.d, p7/m, z11.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n" // c0 (-> z11)
               // --- mul 2
               "ld1d    z31.d, p7/z, %2\n"   // a1 (mul_2)
               "ld1d    z30.d, p7/z, [%8]\n" // b[4]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z12.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z12.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z12.d, p7/m, z12.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // c0 (c1 -> z12)
               // --- mul 3
               "ld1d    z31.d, p7/z, %3\n"   // a2 (mul_3)
               "ld1d    z30.d, p7/z, [%9]\n" // b[8]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z13.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z13.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z13.d, p7/m, z13.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z13.d, p6/m, z13.d, z22.d\n" // c0 (c2 -> z13)
               // --- add 1 (c0 + c1 = z11 + z12)
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z11.d\n"
               "add     z11.d, z12.d, z11.d\n"
               "cmphi   p6.d, p7/z, z12.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n"
               "mov     z30.d, #-4294967295\n" // add_2 ( + c2 = z11 + z13)
               "sub     z30.d, z30.d, z11.d\n"
               "add     z11.d, z13.d, z11.d\n"
               "cmphi   p6.d, p7/z, z13.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n"
               "mov     z16.d, z11.d\n" // r1
               // *** spmv_sve_4x12(r2, a0, a1, a2, &(M[24]));
               "ld1d    z31.d, p7/z, %1\n"    // a0 (mul_1)
               "ld1d    z30.d, p7/z, [%10]\n" // b[0]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z11.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z11.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z11.d, p7/m, z11.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n" // c0 (-> z11)
               // --- mul 2
               "ld1d    z31.d, p7/z, %2\n"    // a1 (mul_2)
               "ld1d    z30.d, p7/z, [%11]\n" // b[4]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z12.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z12.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z12.d, p7/m, z12.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // c0 (c1 -> z12)
               // --- mul 3
               "ld1d    z31.d, p7/z, %3\n"    // a2 (mul_3)
               "ld1d    z30.d, p7/z, [%12]\n" // b[8]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z13.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z13.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z13.d, p7/m, z13.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z13.d, p6/m, z13.d, z22.d\n" // c0 (c2 -> z13)
               // --- add 1 (c0 + c1 = z11 + z12)
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z11.d\n"
               "add     z11.d, z12.d, z11.d\n"
               "cmphi   p6.d, p7/z, z12.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n"
               "mov     z30.d, #-4294967295\n" // add_2 ( + c2 = z11 + z13)
               "sub     z30.d, z30.d, z11.d\n"
               "add     z11.d, z13.d, z11.d\n"
               "cmphi   p6.d, p7/z, z13.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n"
               "mov     z17.d, z11.d\n" // r2
               // *** spmv_sve_4x12(r3, a0, a1, a2, &(M[36]));
               "ld1d    z31.d, p7/z, %1\n"           // a0 (mul_1)
               "ld1d    z30.d, p7/z, [%13]\n"        // b[0]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z11.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z11.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z11.d, p7/m, z11.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n" // c0 (-> z11)
               // --- mul 2
               "ld1d    z31.d, p7/z, %2\n"    // a1 (mul_2)
               "ld1d    z30.d, p7/z, [%14]\n" // b[4]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z12.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z12.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z12.d, p7/m, z12.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // c0 (c1 -> z12)
               // --- mul 3
               "ld1d    z31.d, p7/z, %3\n"    // a2 (mul_3)
               "ld1d    z30.d, p7/z, [%15]\n" // b[8]
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "lsr     z28.d, z30.d, #32\n"         // b_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mov     z24.d, z31.d\n"              // save a_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll = a_l * b_l
               "lsr     z26.d, z31.d, #32\n"         // c_ll_h
               "mad     z30.d, p7/m, z29.d, z26.d\n" // r0 (c_hl)
               "lsr     z26.d, z30.d, #32\n"         // r0_h
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // r0_l
               "mad     z29.d, p7/m, z28.d, z26.d\n" // r2
               "mad     z24.d, p7/m, z28.d, z30.d\n" // r1
               "lsr     z23.d, z24.d, #32\n"         // r1_h
               "add     z23.d, p7/m, z23.d, z29.d\n" // c_h
               "lsl     z25.d, z24.d, #32\n"         // r1_l
               "sel     z29.s, p15, z25.s, z31.s\n"  // c_l
               "lsr     z30.d, z23.d, #32\n"         // c_hh
               "and     z23.d, z23.d, #0xFFFFFFFF\n" // c_hl
               "mov     z27.d, #-4294967295\n"       // GP
               "sub     z28.d, z29.d, z30.d\n"       // c_l - c_hh
               "cmphs   p6.d, p7/z, z29.d, z30.d\n"
               "add     z27.d, z27.d, z28.d\n"
               "sel     z13.d, p6, z28.d, z27.d\n"   // c1
               "mov     z30.d, #-4294967295\n"       // GP
               "sub     z30.d, p7/m, z30.d, z13.d\n" // GP - c1 (GP-a)
               "mul     z23.d, p7/m, z23.d, z22.d\n" // c2 (c_hl * P_n)
               "add     z13.d, p7/m, z13.d, z23.d\n" // c1 + c2
               "cmphi   p6.d, p7/z, z23.d, z30.d\n"
               "add     z13.d, p6/m, z13.d, z22.d\n" // c0 (c2 -> z13)
               // --- add 1 (c0 + c1 = z11 + z12)
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z11.d\n"
               "add     z11.d, z12.d, z11.d\n"
               "cmphi   p6.d, p7/z, z12.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n"
               "mov     z30.d, #-4294967295\n" // add_2 ( + c2 = z11 + z13)
               "sub     z30.d, z30.d, z11.d\n"
               "add     z11.d, z13.d, z11.d\n"
               "cmphi   p6.d, p7/z, z13.d, z30.d\n"
               "add     z11.d, p6/m, z11.d, z22.d\n"
               "mov     z18.d, z11.d\n" // r3
               // *** transpose
#ifdef __SVE_512__
               "ld1d    z11.d, p7/z, [%16]\n" // idx1
               "ld1d    z12.d, p7/z, [%17]\n" // idx2
               "ld1d    z13.d, p7/z, [%18]\n" // idx3
               "ld1d    z14.d, p7/z, [%19]\n" // idx4
               "mov     z19.d, z15.d\n"
               "mov     z20.d, z16.d\n"
               "tbx     z15.d, z17.d, z11.d\n" // t0
               "tbx     z16.d, z18.d, z11.d\n" // t1
               "tbx     z17.d, z19.d, z12.d\n" // t2
               "tbx     z18.d, z20.d, z12.d\n" // t3
               "mov     z19.d, z15.d\n"
               "mov     z20.d, z17.d\n"
               "tbx     z15.d, z16.d, z13.d\n" // c0
               "tbx     z16.d, z19.d, z14.d\n" // c1
               "tbx     z17.d, z18.d, z13.d\n" // c2
               "tbx     z18.d, z20.d, z14.d\n" // c3
#else
               "zip1    z10.d, z15.d, z17.d\n" // t0
               "zip1    z11.d, z16.d, z18.d\n" // t1
               "zip2    z12.d, z15.d, z17.d\n" // t2
               "zip2    z13.d, z16.d, z18.d\n" // t3
               "zip1    z15.d, z10.d, z11.d\n" // c0
               "zip2    z16.d, z10.d, z11.d\n" // c1
               "zip1    z17.d, z12.d, z13.d\n" // c2
               "zip2    z18.d, z12.d, z13.d\n" // c3
#endif
               // add
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z15.d\n"
               "add     z15.d, z16.d, z15.d\n"
               "cmphi   p6.d, p7/z, z16.d, z30.d\n"
               "add     z15.d, p6/m, z15.d, z22.d\n" // sum0
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z17.d\n"
               "add     z17.d, z18.d, z17.d\n"
               "cmphi   p6.d, p7/z, z18.d, z30.d\n"
               "add     z17.d, p6/m, z17.d, z22.d\n" // sum1
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z15.d\n"
               "add     z15.d, z17.d, z15.d\n"
               "cmphi   p6.d, p7/z, z17.d, z30.d\n"
               "add     z15.d, p6/m, z15.d, z22.d\n" // result
               "st1d    z15.d, p7, %0\n"
               : "=m"(b)
               : "m"(a0), "m"(a1), "m"(a2), "r"((uint64_t *)&mm[0]), "r"((uint64_t *)&mm[4]), "r"((uint64_t *)&mm[8]), "r"((uint64_t *)&mm[12]), "r"((uint64_t *)&mm[16]), "r"((uint64_t *)&mm[20]), "r"((uint64_t *)&mm[24]), "r"((uint64_t *)&mm[28]), "r"((uint64_t *)&mm[32]), "r"((uint64_t *)&mm[36]), "r"((uint64_t *)&mm[40]), "r"((uint64_t *)&mm[44])
#ifdef __SVE_512__
                                                                                                                                                                                                                                                                                                                                           ,
                 "r"(idx1), "r"(idx2), "r"(idx3), "r"(idx4), "r"(idx)
#endif
    );
#endif // __USE_SVE_ASM__
}

// Dense matrix-vector product
// We assume coeficients of M_8 can be expressed with 8 bits (<256)
inline void Goldilocks::mmult_sve_4x12_8(svuint64_t &b, const svuint64_t &a0, const svuint64_t &a1, const svuint64_t &a2, const Goldilocks::Element M_8[48])
{
#ifndef __USE_SVE_ASM__
    // Generate matrix 4x4
    svuint64_t r0, r1, r2, r3;
    Goldilocks::spmv_sve_4x12_8(r0, a0, a1, a2, &(M_8[0]));
    Goldilocks::spmv_sve_4x12_8(r1, a0, a1, a2, &(M_8[12]));
    Goldilocks::spmv_sve_4x12_8(r2, a0, a1, a2, &(M_8[24]));
    Goldilocks::spmv_sve_4x12_8(r3, a0, a1, a2, &(M_8[36]));

    // Transpose: transform de 4x4 matrix stored in rows r0...r3 to the columns c0...c3
    svuint64_t c0, c1, c2, c3;
    transpose(r0, r1, r2, r3, c0, c1, c2, c3);

    // Add columns to obtain result
    svuint64_t sum0, sum1;
    add_sve(sum0, c0, c1);
    add_sve(sum1, c2, c3);
    add_sve(b, sum0, sum1);
#else
    uint64_t *mm = (uint64_t *)M_8;
#ifdef __SVE_512__
    uint64_t idx1[8] = {8, 8, 0, 1, 8, 8, 4, 5};
    uint64_t idx2[8] = {2, 3, 8, 8, 6, 7, 8, 8};
    uint64_t idx3[8] = {8, 0, 8, 2, 8, 4, 8, 6};
    uint64_t idx4[8] = {1, 8, 3, 8, 5, 8, 7, 8};
    uint64_t m52[52];
    mm = m52;
    memcpy(mm, M_8, 384);
    uint64_t idx[8] = {8, 8, 8, 8, 0, 1, 2, 3};
#endif
    asm inline("ptrue   p7.d, all\n"
               "mov     z22.d, #4294967295\n" // P_n
                                              // *** spmv_sve_4x12_8(r0, a0, a1, a2, &(M_8[0]));
               "ld1d    z31.d, p7/z, %1\n"    // a0 (mul_1)
               "ld1d    z30.d, p7/z, [%4]\n"
#ifdef __SVE_512__
               "ld1d    z10.d, p7/z, [%20]\n"
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z11.d, z29.d, #32\n"         // c_h (c0_h -> z11)
               "ptrue   p15.s, all\n"
               "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z12.s, p15, z30.s, z31.s\n"  // c_l (c0_l -> z12)
               "ld1d    z31.d, p7/z, %2\n"           // a1  (mul_2)
               "ld1d    z30.d, p7/z, [%5]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z13.d, z29.d, #32\n"         // c_h (c1_h -> z13)
               "sel     z14.s, p15, z30.s, z31.s\n"  // c_l (c1_l -> z14)
               "ld1d    z31.d, p7/z, %3\n"           // a2  (mul_3)
               "ld1d    z30.d, p7/z, [%6]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z15.d, z29.d, #32\n"         // c_h (c2_h -> z15)
               "sel     z16.s, p15, z30.s, z31.s\n"  // c_l (c2_l -> z16)
               "mov     z30.d, #-4294967295\n"       // add_sve (z12 + z14)
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z14.d, z12.d\n"
               "cmphi   p6.d, p7/z, z14.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // aux_l -> z12
               "mov     z30.d, #-4294967295\n"       // add_sve (z12 + z16)
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z16.d, z12.d\n"
               "cmphi   p6.d, p7/z, z16.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // c_l -> z12
               "add     z11.d, p7/m, z11.d, z13.d\n"
               "add     z11.d, p7/m, z11.d, z15.d\n" // c_h -> z11
               "mul     z11.d, p7/m, z11.d, z22.d\n" // c1
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z11.d, z12.d\n"
               "cmphi   p6.d, p7/z, z11.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n"
               "mov     z17.d, z12.d\n" // r0
               // *** spmv_sve_4x12_8(r1, a0, a1, a2, &(M_8[12]));
               "ld1d    z31.d, p7/z, %1\n" // a0 (mul_1)
               "ld1d    z30.d, p7/z, [%7]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z11.d, z29.d, #32\n"         // c_h (c0_h -> z11)
               // "ptrue   p15.s, all\n"
               // "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z12.s, p15, z30.s, z31.s\n" // c_l (c0_l -> z12)
               "ld1d    z31.d, p7/z, %2\n"          // a1  (mul_2)
               "ld1d    z30.d, p7/z, [%8]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z13.d, z29.d, #32\n"         // c_h (c1_h -> z13)
               "sel     z14.s, p15, z30.s, z31.s\n"  // c_l (c1_l -> z14)
               "ld1d    z31.d, p7/z, %3\n"           // a2  (mul_3)
               "ld1d    z30.d, p7/z, [%9]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z15.d, z29.d, #32\n"         // c_h (c2_h -> z15)
               "sel     z16.s, p15, z30.s, z31.s\n"  // c_l (c2_l -> z16)
               "mov     z30.d, #-4294967295\n"       // add_sve (z12 + z14)
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z14.d, z12.d\n"
               "cmphi   p6.d, p7/z, z14.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // aux_l -> z12
               "mov     z30.d, #-4294967295\n"       // add_sve (z12 + z16)
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z16.d, z12.d\n"
               "cmphi   p6.d, p7/z, z16.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // c_l -> z12
               "add     z11.d, p7/m, z11.d, z13.d\n"
               "add     z11.d, p7/m, z11.d, z15.d\n" // c_h -> z11
               "mul     z11.d, p7/m, z11.d, z22.d\n" // c1
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z11.d, z12.d\n"
               "cmphi   p6.d, p7/z, z11.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n"
               "mov     z18.d, z12.d\n" // r1
               // *** spmv_sve_4x12_8(r2, a0, a1, a2, &(M_8[24]));
               "ld1d    z31.d, p7/z, %1\n" // a0 (mul_1)
               "ld1d    z30.d, p7/z, [%10]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z11.d, z29.d, #32\n"         // c_h (c0_h -> z11)
               // "ptrue   p15.s, all\n"
               // "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z12.s, p15, z30.s, z31.s\n" // c_l (c0_l -> z12)
               "ld1d    z31.d, p7/z, %2\n"          // a1  (mul_2)
               "ld1d    z30.d, p7/z, [%11]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z13.d, z29.d, #32\n"         // c_h (c1_h -> z13)
               "sel     z14.s, p15, z30.s, z31.s\n"  // c_l (c1_l -> z14)
               "ld1d    z31.d, p7/z, %3\n"           // a2  (mul_3)
               "ld1d    z30.d, p7/z, [%12]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z15.d, z29.d, #32\n"         // c_h (c2_h -> z15)
               "sel     z16.s, p15, z30.s, z31.s\n"  // c_l (c2_l -> z16)
               "mov     z30.d, #-4294967295\n"       // add_sve (z12 + z14)
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z14.d, z12.d\n"
               "cmphi   p6.d, p7/z, z14.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // aux_l -> z12
               "mov     z30.d, #-4294967295\n"       // add_sve (z12 + z16)
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z16.d, z12.d\n"
               "cmphi   p6.d, p7/z, z16.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // c_l -> z12
               "add     z11.d, p7/m, z11.d, z13.d\n"
               "add     z11.d, p7/m, z11.d, z15.d\n" // c_h -> z11
               "mul     z11.d, p7/m, z11.d, z22.d\n" // c1
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z11.d, z12.d\n"
               "cmphi   p6.d, p7/z, z11.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n"
               "mov     z19.d, z12.d\n" // r2
               // *** spmv_sve_4x12_8(r3, a0, a1, a2, &(M_8[36]));
               "ld1d    z31.d, p7/z, %1\n" // a0 (mul_1)
               "ld1d    z30.d, p7/z, [%13]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z11.d, z29.d, #32\n"         // c_h (c0_h -> z11)
               // "ptrue   p15.s, all\n"
               // "eor     p15.b, p15/z, p7.b, p15.b\n" // sel
               "sel     z12.s, p15, z30.s, z31.s\n" // c_l (c0_l -> z12)
               "ld1d    z31.d, p7/z, %2\n"          // a1  (mul_2)
               "ld1d    z30.d, p7/z, [%14]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z13.d, z29.d, #32\n"         // c_h (c1_h -> z13)
               "sel     z14.s, p15, z30.s, z31.s\n"  // c_l (c1_l -> z14)
               "ld1d    z31.d, p7/z, %3\n"           // a2  (mul_3)
               "ld1d    z30.d, p7/z, [%15]\n"
#ifdef __SVE_512__
               "tbx     z30.d, z30.d, z10.d\n"
#endif
               "lsr     z29.d, z31.d, #32\n"         // a_h
               "and     z31.d, z31.d, #0xFFFFFFFF\n" // a_l
               "and     z30.d, z30.d, #0xFFFFFFFF\n" // b_l
               "mul     z31.d, p7/m, z31.d, z30.d\n" // c_ll
               "lsr     z28.d, z31.d, #32\n"         // c_ll_h
               "mad     z29.d, p7/m, z30.d, z28.d\n" // r0
               "lsl     z30.d, z29.d, #32\n"         // r0_l
               "lsr     z15.d, z29.d, #32\n"         // c_h (c2_h -> z15)
               "sel     z16.s, p15, z30.s, z31.s\n"  // c_l (c2_l -> z16)
               "mov     z30.d, #-4294967295\n"       // add_sve (z12 + z14)
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z14.d, z12.d\n"
               "cmphi   p6.d, p7/z, z14.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // aux_l -> z12
               "mov     z30.d, #-4294967295\n"       // add_sve (z12 + z16)
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z16.d, z12.d\n"
               "cmphi   p6.d, p7/z, z16.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n" // c_l -> z12
               "add     z11.d, p7/m, z11.d, z13.d\n"
               "add     z11.d, p7/m, z11.d, z15.d\n" // c_h -> z11
               "mul     z11.d, p7/m, z11.d, z22.d\n" // c1
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z12.d\n"
               "add     z12.d, z11.d, z12.d\n"
               "cmphi   p6.d, p7/z, z11.d, z30.d\n"
               "add     z12.d, p6/m, z12.d, z22.d\n"
               "mov     z20.d, z12.d\n" // r3
               // *** transpose
#ifdef __SVE_512__
               "ld1d    z11.d, p7/z, [%16]\n" // idx1
               "ld1d    z12.d, p7/z, [%17]\n" // idx2
               "ld1d    z13.d, p7/z, [%18]\n" // idx3
               "ld1d    z14.d, p7/z, [%19]\n" // idx4
               "mov     z15.d, z17.d\n"
               "mov     z16.d, z18.d\n"
               "tbx     z15.d, z19.d, z11.d\n" // t0
               "tbx     z16.d, z20.d, z11.d\n" // t1
               "tbx     z19.d, z17.d, z12.d\n" // t2
               "tbx     z20.d, z18.d, z12.d\n" // t3
               "mov     z17.d, z19.d\n"
               "mov     z18.d, z20.d\n"
               "mov     z10.d, z15.d\n"
               "tbx     z15.d, z16.d, z13.d\n" // c0
               "tbx     z16.d, z10.d, z14.d\n" // c1
               "tbx     z17.d, z18.d, z13.d\n" // c2
               "tbx     z18.d, z19.d, z14.d\n" // c3
#else
               "zip1    z10.d, z17.d, z19.d\n" // t0
               "zip1    z11.d, z18.d, z20.d\n" // t1
               "zip2    z12.d, z17.d, z19.d\n" // t2
               "zip2    z13.d, z18.d, z20.d\n" // t3
               "zip1    z15.d, z10.d, z11.d\n" // c0
               "zip2    z16.d, z10.d, z11.d\n" // c1
               "zip1    z17.d, z12.d, z13.d\n" // c2
               "zip2    z18.d, z12.d, z13.d\n" // c3
#endif
               // add
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z15.d\n"
               "add     z15.d, z16.d, z15.d\n"
               "cmphi   p6.d, p7/z, z16.d, z30.d\n"
               "add     z15.d, p6/m, z15.d, z22.d\n" // sum0
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z17.d\n"
               "add     z17.d, z18.d, z17.d\n"
               "cmphi   p6.d, p7/z, z18.d, z30.d\n"
               "add     z17.d, p6/m, z17.d, z22.d\n" // sum1
               "mov     z30.d, #-4294967295\n"
               "sub     z30.d, z30.d, z15.d\n"
               "add     z15.d, z17.d, z15.d\n"
               "cmphi   p6.d, p7/z, z17.d, z30.d\n"
               "add     z15.d, p6/m, z15.d, z22.d\n" // result
               "st1d    z15.d, p7, %0\n"
               : "=m"(b)
               : "m"(a0), "m"(a1), "m"(a2), "r"((uint64_t *)&mm[0]), "r"((uint64_t *)&mm[4]), "r"((uint64_t *)&mm[8]), "r"((uint64_t *)&mm[12]), "r"((uint64_t *)&mm[16]), "r"((uint64_t *)&mm[20]), "r"((uint64_t *)&mm[24]), "r"((uint64_t *)&mm[28]), "r"((uint64_t *)&mm[32]), "r"((uint64_t *)&mm[36]), "r"((uint64_t *)&mm[40]), "r"((uint64_t *)&mm[44])
#ifdef __SVE_512__
                                                                                                                                                                                                                                                                                                                                           ,
                 "r"(idx1), "r"(idx2), "r"(idx3), "r"(idx4), "r"(idx)
#endif
               );
#endif // __USE_SVE_ASM__
}

inline void Goldilocks::mmult_sve(svuint64_t &a0, svuint64_t &a1, svuint64_t &a2, const Goldilocks::Element M[144])
{
    svuint64_t b0, b1, b2;
    Goldilocks::mmult_sve_4x12(b0, a0, a1, a2, &(M[0]));
    Goldilocks::mmult_sve_4x12(b1, a0, a1, a2, &(M[48]));
    Goldilocks::mmult_sve_4x12(b2, a0, a1, a2, &(M[96]));
    a0 = b0;
    a1 = b1;
    a2 = b2;
}

// We assume coeficients of M_8 can be expressed with 8 bits (<256)
inline void Goldilocks::mmult_sve_8(svuint64_t &a0, svuint64_t &a1, svuint64_t &a2, const Goldilocks::Element M_8[144])
{
    svuint64_t b0, b1, b2;
    Goldilocks::mmult_sve_4x12_8(b0, a0, a1, a2, &(M_8[0]));
    Goldilocks::mmult_sve_4x12_8(b1, a0, a1, a2, &(M_8[48]));
    Goldilocks::mmult_sve_4x12_8(b2, a0, a1, a2, &(M_8[96]));
    a0 = b0;
    a1 = b1;
    a2 = b2;
}

/*
    Implementations for expressions:
*/
inline void Goldilocks::copy_sve(Element *dst, const Element &src)
{
    // Does not make sense to vectorize yet
    for (uint64_t i = 0; i < SVE_SIZE_; ++i)
    {
        dst[i].fe = src.fe;
    }
}

inline void Goldilocks::copy_sve(Element *dst, const Element *src)
{
    // Does not make sense to vectorize yet
    for (uint64_t i = 0; i < SVE_SIZE_; ++i)
    {
        dst[i].fe = src[i].fe;
    }
}

inline void Goldilocks::copy_sve(Element *dst, const Element *src, uint64_t stride)
{
    // Does not make sense to vectorize yet
    for (uint64_t i = 0; i < SVE_SIZE_; ++i)
    {
        dst[i].fe = src[i * stride].fe;
    }
}

inline void Goldilocks::copy_sve(Element *dst, uint64_t stride_dst, const Element *src, uint64_t stride)
{
    // Does not make sense to vectorize yet
    for (uint64_t i = 0; i < SVE_SIZE_; ++i)
    {
        dst[i * stride_dst].fe = src[i * stride].fe;
    }
}

inline void Goldilocks::copy_sve(Element *dst, const Element *src, uint64_t stride[4])
{
    // Does not make sense to vectorize yet
    for (uint64_t i = 0; i < SVE_SIZE_; ++i)
    {
        dst[i].fe = src[stride[i]].fe;
    }
}

inline void Goldilocks::copy_sve(svuint64_t &dst_, const Element &src)
{
    Element dst[4];
    for (uint64_t i = 0; i < SVE_SIZE_; ++i)
    {
        dst[i].fe = src.fe;
    }
    load_sve(dst_, dst);
}

inline void Goldilocks::copy_sve(svuint64_t &dst_, const svuint64_t &src_)
{
    dst_ = src_;
}

inline void Goldilocks::copy_sve(svuint64_t &dst_, const Element *src, uint64_t stride)
{
    Element dst[4];
    for (uint64_t i = 0; i < SVE_SIZE_; ++i)
    {
        dst[i].fe = src[i * stride].fe;
    }
    load_sve(dst_, dst);
}

inline void Goldilocks::copy_sve(svuint64_t &dst_, const Element *src, uint64_t stride[4])
{
    Element dst[4];
    for (uint64_t i = 0; i < SVE_SIZE_; ++i)
    {
        dst[i].fe = src[stride[i]].fe;
    }
    load_sve(dst_, dst);
};

inline void Goldilocks::copy_sve(Element *dst, uint64_t stride, const svuint64_t &src_)
{
    Element src[4];
    Goldilocks::store_sve(src, src_);
    dst[0] = src[0];
    dst[stride] = src[1];
    dst[2 * stride] = src[2];
    dst[3 * stride] = src[3];
}

inline void Goldilocks::copy_sve(Element *dst, uint64_t stride[4], const svuint64_t &src_)
{
    Element src[4];
    Goldilocks::store_sve(src, src_);
    for (uint64_t i = 0; i < SVE_SIZE_; ++i)
    {
        dst[stride[i]].fe = src[i].fe;
    }
}

inline void Goldilocks::add_sve(Element *c4, const Element *a4, const Element *b4)
{
    svuint64_t a_, b_, c_;
    load_sve(a_, a4);
    load_sve(b_, b4);
    add_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::add_sve(Element *c4, const Element *a4, const Element *b4, uint64_t offset_b)
{
    Element bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b4[k * offset_b];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, a4);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::add_sve(Element *c4, const Element *a4, const Element *b4, const uint64_t offset_b[4])
{
    Element bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b4[offset_b[k]];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, a4);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::add_sve(Element *c4, const Element *a4, const Element b)
{
    Element bb[4] = {b, b, b, b};
    svuint64_t a_, b_, c_;
    load_sve(a_, a4);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::add_sve(Element *c4, const Element *a4, const Element b, uint64_t offset_a)
{
    Element bb[4] = {b, b, b, b};
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::add_sve(Element *c4, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b)
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
        bb[k] = b4[k * offset_b];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::add_sve(Element *c4, uint64_t offset_c, const Element *a4, uint64_t offset_a, const Element *b4, uint64_t offset_b)
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
        bb[k] = b4[k * offset_b];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);

    Element cc[4];
    store_sve(cc, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c4[k * offset_c] = cc[k];
    }
}

inline void Goldilocks::add_sve(Element *c4, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4])
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
        bb[k] = b4[offset_b[k]];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::add_sve(Element *c4, const Element *a4, const Element b, const uint64_t offset_a[4])
{
    Element bb[4] = {b, b, b, b};
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::add_sve(svuint64_t &c_, const svuint64_t &a_, const Element *b4, uint64_t offset_b)
{
    Element bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b4[k * offset_b];
    }
    svuint64_t b_;
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
};

inline void Goldilocks::add_sve(Element *c, uint64_t offset_c, const svuint64_t &a_, const svuint64_t &b_)
{
    svuint64_t c_;
    add_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[k * offset_c] = c4[k];
    }
}

inline void Goldilocks::add_sve(Element *c, uint64_t offset_c, const svuint64_t &a_, const Element *b, uint64_t offset_b)
{
    Element bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b[k * offset_b];
    }
    svuint64_t b_, c_;
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[k * offset_c] = c4[k];
    }
}

inline void Goldilocks::add_sve(Element *c, const uint64_t offset_c[4], const svuint64_t &a_, const svuint64_t &b_)
{
    svuint64_t c_;
    add_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[offset_c[k]] = c4[k];
    }
}

inline void Goldilocks::add_sve(Element *c, const uint64_t offset_c[4], const svuint64_t &a_, const Element *b, uint64_t offset_b)
{
    Element b4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        b4[k] = b[k * offset_b];
    }
    svuint64_t b_, c_;
    load_sve(b_, b4);
    add_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[offset_c[k]] = c4[k];
    }
}

inline void Goldilocks::add_sve(Element *c, const uint64_t offset_c[4], const svuint64_t &a_, const Element *b, uint64_t offset_b[4])
{
    Element b4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        b4[k] = b[offset_b[k]];
    }
    svuint64_t b_, c_;
    load_sve(b_, b4);
    add_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[offset_c[k]] = c4[k];
    }
}

inline void Goldilocks::add_sve(svuint64_t &c_, const svuint64_t &a_, const Element *b4, const uint64_t offset_b[4])
{
    Element bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b4[offset_b[k]];
    }
    svuint64_t b_;
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
};

inline void Goldilocks::add_sve(svuint64_t &c_, const svuint64_t &a_, const Element b)
{
    Element bb[4] = {b, b, b, b};
    svuint64_t b_;
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
};

inline void Goldilocks::add_sve(svuint64_t &c_, const Element *a4, const Element b, uint64_t offset_a)
{
    Element bb[4] = {b, b, b, b};
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
    }
    svuint64_t a_, b_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
};

inline void Goldilocks::add_sve(svuint64_t &c_, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b)
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
        bb[k] = b4[k * offset_b];
    }
    svuint64_t a_, b_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
};

inline void Goldilocks::add_sve(svuint64_t &c_, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4])
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
        bb[k] = b4[offset_b[k]];
    }
    svuint64_t a_, b_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
};

inline void Goldilocks::add_sve(svuint64_t &c_, const Element *a4, const Element b, const uint64_t offset_a[4])
{

    Element bb[4] = {b, b, b, b};
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
    }
    svuint64_t a_, b_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    add_sve(c_, a_, b_);
};

inline void Goldilocks::sub_sve(Goldilocks::Element *c4, const Goldilocks::Element *a4, const Goldilocks::Element *b4)
{
    svuint64_t a_, b_, c_;
    load_sve(a_, a4);
    load_sve(b_, b4);
    sub_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::sub_sve(Element *c4, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b)
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
        bb[k] = b4[k * offset_b];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    sub_sve(c_, a_, b_);
    store_sve(c4, c_);
};

inline void Goldilocks::sub_sve(Element *c4, const Element *a4, const Element b)
{
    svuint64_t a_, b_, c_;
    Goldilocks::Element b4[4] = {b, b, b, b};
    load_sve(a_, a4);
    load_sve(b_, b4);
    sub_sve(c_, a_, b_);
    store_sve(c4, c_);
};

inline void Goldilocks::sub_sve(Element *c4, const Element a, const Element *b4)
{
    svuint64_t a_, b_, c_;
    Goldilocks::Element a4[4] = {a, a, a, a};
    load_sve(a_, a4);
    load_sve(b_, b4);
    sub_sve(c_, a_, b_);
    store_sve(c4, c_);
};

inline void Goldilocks::sub_sve(Element *c4, const Element *a4, const Element b, uint64_t offset_a)
{
    svuint64_t a_, b_, c_;
    Goldilocks::Element b4[4] = {b, b, b, b};
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
    }
    load_sve(a_, aa);
    load_sve(b_, b4);
    sub_sve(c_, a_, b_);
    store_sve(c4, c_);
};

inline void Goldilocks::sub_sve(Element *c4, const Element a, const Element *b4, uint64_t offset_b)
{
    svuint64_t a_, b_, c_;
    Goldilocks::Element a4[4], bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k].fe = b4[k * offset_b].fe;
        a4[k].fe = a.fe;
    }
    load_sve(a_, a4);
    load_sve(b_, bb);
    sub_sve(c_, a_, b_);
    store_sve(c4, c_);
};

inline void Goldilocks::sub_sve(Element *c4, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4])
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
        bb[k] = b4[offset_b[k]];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    sub_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::sub_sve(Element *c4, const Element a, const Element *b4, const uint64_t offset_b[4])
{
    svuint64_t a_, b_, c_;
    Goldilocks::Element a4[4], bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k].fe = b4[offset_b[k]].fe;
        a4[k].fe = a.fe;
    }
    load_sve(a_, a4);
    load_sve(b_, bb);
    sub_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::sub_sve(Element *c4, const Element *a4, const Element b, const uint64_t offset_a[4])
{
    svuint64_t a_, b_, c_;
    Goldilocks::Element b4[4] = {b, b, b, b};
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
    }
    load_sve(a_, aa);
    load_sve(b_, b4);
    sub_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b)
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
        bb[k] = b4[k * offset_b];
    }
    svuint64_t a_, b_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const svuint64_t &a_, const Element *b4, uint64_t offset_b)
{
    Element bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b4[k * offset_b];
    }
    svuint64_t b_;
    load_sve(b_, bb);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const Element *a4, uint64_t offset_a, const svuint64_t &b_)
{
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
    }
    svuint64_t a_;
    load_sve(a_, aa);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const Element *a4, const svuint64_t &b_, uint64_t offset_a)
{
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
    }
    svuint64_t a_;
    load_sve(a_, aa);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const svuint64_t &a_, const Element b)
{
    svuint64_t b_;
    Goldilocks::Element b4[4] = {b, b, b, b};
    load_sve(b_, b4);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const Element a, const svuint64_t &b_)
{
    svuint64_t a_;
    Goldilocks::Element a4[4] = {a, a, a, a};
    load_sve(a_, a4);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const Element *a4, const Element b, uint64_t offset_a)
{
    svuint64_t a_, b_;
    Goldilocks::Element b4[4] = {b, b, b, b};
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
    }
    load_sve(a_, aa);
    load_sve(b_, b4);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const Element a, const Element *b4, uint64_t offset_b)
{
    svuint64_t a_, b_;
    Goldilocks::Element a4[4], bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b4[k * offset_b];
        a4[k] = a;
    }
    load_sve(a_, a4);
    load_sve(b_, bb);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4])
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
        bb[k] = b4[offset_b[k]];
    }
    svuint64_t a_, b_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const Element a, const Element *b4, const uint64_t offset_b[4])
{
    svuint64_t a_, b_;
    Goldilocks::Element a4[4], bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b4[offset_b[k]];
        a4[k] = a;
    }
    load_sve(a_, a4);
    load_sve(b_, bb);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const Element *a4, const Element b, const uint64_t offset_a[4])
{
    svuint64_t a_, b_;
    Goldilocks::Element b4[4] = {b, b, b, b};
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
    }
    load_sve(a_, aa);
    load_sve(b_, b4);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const svuint64_t &a_, const Element *b4, uint64_t offset_b[4])
{
    Element bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b4[offset_b[k]];
    }
    svuint64_t b_;
    load_sve(b_, bb);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(svuint64_t &c_, const Element *a4, const svuint64_t &b_, uint64_t offset_a[4])
{
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
    }
    svuint64_t a_;
    load_sve(a_, aa);
    sub_sve(c_, a_, b_);
}

inline void Goldilocks::sub_sve(Element *c, uint64_t offset_c, const svuint64_t &a_, const svuint64_t &b_)
{
    svuint64_t c_;
    sub_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[k * offset_c] = c4[k];
    }
}

inline void Goldilocks::sub_sve(Element *c, const uint64_t offset_c[4], const svuint64_t &a_, const svuint64_t &b_)
{
    svuint64_t c_;
    sub_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[offset_c[k]] = c4[k];
    }
}

inline void Goldilocks::sub_sve(Element *c, uint64_t offset_c, const Element a, const svuint64_t &b_)
{
    svuint64_t a_, c_;
    Goldilocks::Element a4[4] = {a, a, a, a};
    load_sve(a_, a4);
    sub_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[k * offset_c] = c4[k];
    }
}

inline void Goldilocks::sub_sve(Element *c, const uint64_t offset_c[4], const Element a, const svuint64_t &b_)
{
    svuint64_t a_, c_;
    Goldilocks::Element a4[4] = {a, a, a, a};
    load_sve(a_, a4);
    sub_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[offset_c[k]] = c4[k];
    }
}

inline void Goldilocks::mul_sve(Element *c4, const Element *a4, const Element *b4)
{
    svuint64_t a_, b_, c_;
    load_sve(a_, a4);
    load_sve(b_, b4);
    mult_sve(c_, a_, b_);
    store_sve(c4, c_);
};

inline void Goldilocks::mul_sve(Element *c4, const Element a, const Element *b4)
{
    svuint64_t a_, b_, c_;
    Goldilocks::Element a4[4] = {a, a, a, a};
    load_sve(a_, a4);
    load_sve(b_, b4);
    mult_sve(c_, a_, b_);
    store_sve(c4, c_);
};

inline void Goldilocks::mul_sve(Element *c4, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b)
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
        bb[k] = b4[k * offset_b];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    mult_sve(c_, a_, b_);
    store_sve(c4, c_);
};

inline void Goldilocks::mul_sve(Element *c4, const Element a, const Element *b4, uint64_t offset_b)
{
    svuint64_t a_, b_, c_;
    Goldilocks::Element a4[4], bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k].fe = b4[k * offset_b].fe;
        a4[k].fe = a.fe;
    }
    load_sve(a_, a4);
    load_sve(b_, bb);
    mult_sve(c_, a_, b_);
    store_sve(c4, c_);
};

inline void Goldilocks::mul_sve(Element *c4, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4])
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
        bb[k] = b4[offset_b[k]];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    mult_sve(c_, a_, b_);
    store_sve(c4, c_);
}

inline void Goldilocks::mul_sve(svuint64_t &c_, const Element a, const svuint64_t &b_)
{
    svuint64_t a_;
    Goldilocks::Element a4[4] = {a, a, a, a};
    load_sve(a_, a4);
    mult_sve(c_, a_, b_);
};

inline void Goldilocks::mul_sve(svuint64_t &c_, const Element *a4, const Element *b4, uint64_t offset_a, uint64_t offset_b)
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
        bb[k] = b4[k * offset_b];
    }
    svuint64_t a_, b_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    mult_sve(c_, a_, b_);
};

inline void Goldilocks::mul_sve(svuint64_t &c_, const svuint64_t &a_, const Element *b4, uint64_t offset_b)
{
    Element bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b4[k * offset_b];
    }
    svuint64_t b_;
    load_sve(b_, bb);
    mult_sve(c_, a_, b_);
}

inline void Goldilocks::mul_sve(svuint64_t &c_, const Element *a4, const svuint64_t &b_, uint64_t offset_a)
{
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[k * offset_a];
    }
    svuint64_t a_;
    load_sve(a_, aa);
    mult_sve(c_, a_, b_);
}

inline void Goldilocks::mul_sve(svuint64_t &c_, const Element a, const Element *b4, uint64_t offset_b)
{
    svuint64_t a_, b_;
    Goldilocks::Element aa[4], bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k].fe = b4[k * offset_b].fe;
        aa[k].fe = a.fe;
    }
    load_sve(a_, aa);
    load_sve(b_, bb);
    mult_sve(c_, a_, b_);
};

inline void Goldilocks::mul_sve(svuint64_t &c_, const Element *a4, const Element *b4, const uint64_t offset_a[4], const uint64_t offset_b[4])
{
    Element bb[4];
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
        bb[k] = b4[offset_b[k]];
    }
    svuint64_t a_, b_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    mult_sve(c_, a_, b_);
};

inline void Goldilocks::mul_sve(svuint64_t &c_, const svuint64_t &a_, const Element *b4, const uint64_t offset_b[4])
{
    Element bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        bb[k] = b4[offset_b[k]];
    }
    svuint64_t b_;
    load_sve(b_, bb);
    mult_sve(c_, a_, b_);
}

inline void Goldilocks::mul_sve(svuint64_t &c_, const Element *a4, const svuint64_t &b_, const uint64_t offset_a[4])
{
    Element aa[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
    }
    svuint64_t a_;
    load_sve(a_, aa);
    mult_sve(c_, a_, b_);
}

inline void Goldilocks::mul_sve(Element *c, uint64_t offset_c[4], const Element *a, const svuint64_t &b_, const uint64_t offset_a[4])
{
    Element a4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        a4[k] = a[offset_a[k]];
    }
    svuint64_t a_, c_;
    load_sve(a_, a4);
    mult_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[offset_c[k]] = c4[k];
    }
};

inline void Goldilocks::mul_sve(Element *c, uint64_t offset_c[4], const Element *a, const Element *b, const uint64_t offset_a[4], const uint64_t offset_b[4])
{
    Element a4[4];
    Element b4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        a4[k] = a[offset_a[k]];
        b4[k] = b[offset_b[k]];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, a4);
    load_sve(b_, b4);
    mult_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[offset_c[k]] = c4[k];
    }
};

inline void Goldilocks::mul_sve(svuint64_t &c_, const Element *a4, const Element b, const uint64_t offset_a[4])
{
    Element aa[4], bb[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        aa[k] = a4[offset_a[k]];
        bb[k] = b;
    }
    svuint64_t a_, b_;
    load_sve(a_, aa);
    load_sve(b_, bb);
    mult_sve(c_, a_, b_);
}

inline void Goldilocks::mul_sve(Element *c, uint64_t offset_c, const svuint64_t &a_, const svuint64_t &b_)
{
    svuint64_t c_;
    mult_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[k * offset_c] = c4[k];
    }
};

inline void Goldilocks::mul_sve(Element *c, uint64_t offset_c, const Element *a, const svuint64_t &b_, uint64_t offset_a)
{
    Element a4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        a4[k] = a[k * offset_a];
    }
    svuint64_t a_, c_;
    load_sve(a_, a4);
    mult_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[k * offset_c] = c4[k];
    }
};

inline void Goldilocks::mul_sve(Element *c, uint64_t offset_c, const svuint64_t &a_, const Element *b, uint64_t offset_b)
{

    Element b4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        b4[k] = b[k * offset_b];
    }
    svuint64_t b_, c_;
    load_sve(b_, b4);
    mult_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[k * offset_c] = c4[k];
    }
};

inline void Goldilocks::mul_sve(Element *c, uint64_t offset_c, const Element *a, uint64_t offset_a, const Element *b, uint64_t offset_b)
{
    Element b4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        b4[k] = b[k * offset_b];
    }
    Element a4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        a4[k] = a[k * offset_a];
    }
    svuint64_t a_, b_, c_;
    load_sve(a_, a4);
    load_sve(b_, b4);
    mult_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[k * offset_c] = c4[k];
    }
};

inline void Goldilocks::mul_sve(Element *c, uint64_t offset_c, const Element *a, const svuint64_t &b_, const uint64_t offset_a[4])
{
    Element a4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        a4[k] = a[offset_a[k]];
    }
    svuint64_t a_, c_;
    load_sve(a_, a4);
    mult_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[k * offset_c] = c4[k];
    }
};

inline void Goldilocks::mul_sve(Element *c, uint64_t offset_c[4], const svuint64_t &a_, const svuint64_t &b_)
{
    svuint64_t c_;
    mult_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[offset_c[k]] = c4[k];
    }
};

inline void Goldilocks::mul_sve(Element *c, uint64_t offset_c[4], const Element *a, const svuint64_t &b_, uint64_t offset_a)
{
    Element a4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        a4[k] = a[k * offset_a];
    }
    svuint64_t a_, c_;
    load_sve(a_, a4);
    mult_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[offset_c[k]] = c4[k];
    }
};

inline void Goldilocks::mul_sve(Element *c, uint64_t offset_c[4], const svuint64_t &a_, const Element *b, uint64_t offset_b)
{
    Element b4[4];
    for (uint64_t k = 0; k < 4; ++k)
    {
        b4[k] = b[k * offset_b];
    }
    svuint64_t b_, c_;
    load_sve(b_, b4);
    mult_sve(c_, a_, b_);
    Element c4[4];
    store_sve(c4, c_);
    for (uint64_t k = 0; k < 4; ++k)
    {
        c[offset_c[k]] = c4[k];
    }
};
#endif
