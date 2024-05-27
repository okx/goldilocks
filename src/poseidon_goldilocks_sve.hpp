#ifndef POSEIDON_GOLDILOCKS_SVE
#define POSEIDON_GOLDILOCKS_SVE

#include "poseidon_goldilocks.hpp"
#include "goldilocks_base_field.hpp"

inline void PoseidonGoldilocks::hash(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

inline void PoseidonGoldilocks::pow7_sve(svuint64_t &st0, svuint64_t &st1, svuint64_t &st2)
{
    svuint64_t pw2_0, pw2_1, pw2_2;
    Goldilocks::square_sve(pw2_0, st0);
    Goldilocks::square_sve(pw2_1, st1);
    Goldilocks::square_sve(pw2_2, st2);
    svuint64_t pw4_0, pw4_1, pw4_2;
    Goldilocks::square_sve(pw4_0, pw2_0);
    Goldilocks::square_sve(pw4_1, pw2_1);
    Goldilocks::square_sve(pw4_2, pw2_2);
    svuint64_t pw3_0, pw3_1, pw3_2;
    Goldilocks::mult_sve(pw3_0, pw2_0, st0);
    Goldilocks::mult_sve(pw3_1, pw2_1, st1);
    Goldilocks::mult_sve(pw3_2, pw2_2, st2);

    Goldilocks::mult_sve(st0, pw3_0, pw4_0);
    Goldilocks::mult_sve(st1, pw3_1, pw4_1);
    Goldilocks::mult_sve(st2, pw3_2, pw4_2);
};

inline void PoseidonGoldilocks::add_sve(svuint64_t &st0, svuint64_t &st1, svuint64_t &st2, const Goldilocks::Element C_[SPONGE_WIDTH])
{
    svuint64_t c0, c1, c2;
    Goldilocks::load_sve(c0, &(C_[0]));
    Goldilocks::load_sve(c1, &(C_[4]));
    Goldilocks::load_sve(c2, &(C_[8]));
    Goldilocks::add_sve(st0, st0, c0);
    Goldilocks::add_sve(st1, st1, c1);
    Goldilocks::add_sve(st2, st2, c2);
}
// Assuming C_a is aligned
inline void PoseidonGoldilocks::add_sve_a(svuint64_t &st0, svuint64_t &st1, svuint64_t &st2, const Goldilocks::Element C_a[SPONGE_WIDTH])
{
    svuint64_t c0, c1, c2;
    Goldilocks::load_sve_a(c0, &(C_a[0]));
    Goldilocks::load_sve_a(c1, &(C_a[4]));
    Goldilocks::load_sve_a(c2, &(C_a[8]));
    Goldilocks::add_sve(st0, st0, c0);
    Goldilocks::add_sve(st1, st1, c1);
    Goldilocks::add_sve(st2, st2, c2);
}
inline void PoseidonGoldilocks::add_sve_small(svuint64_t &st0, svuint64_t &st1, svuint64_t &st2, const Goldilocks::Element C_small[SPONGE_WIDTH])
{
    svuint64_t c0, c1, c2;
    Goldilocks::load_sve(c0, &(C_small[0]));
    Goldilocks::load_sve(c1, &(C_small[4]));
    Goldilocks::load_sve(c2, &(C_small[8]));

    Goldilocks::add_sve(st0, st0, c0);
    Goldilocks::add_sve(st1, st1, c1);
    Goldilocks::add_sve(st2, st2, c2);
}

#endif      // POSEIDON_GOLDILOCKS_SVE