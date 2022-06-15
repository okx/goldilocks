#ifndef POSEIDON_GOLDILOCKS
#define POSEIDON_GOLDILOCKS

#include "poseidon_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"

#define RATE 8
#define CAPACITY 4
#define SPONGE_WIDTH (RATE + CAPACITY)
#define HALF_N_FULL_ROUNDS 4
#define N_FULL_ROUNDS_TOTAL (2 * HALF_N_FULL_ROUNDS)
#define N_PARTIAL_ROUNDS 22
#define N_ROUNDS (N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS)

class PoseidonGoldilocks
{

private:
    inline void static pow7(Goldilocks::Element &x)
    {
        Goldilocks::Element x2 = x * x;
        Goldilocks::Element x3 = x * x2;
        Goldilocks::Element x4 = x2 * x2;
        x = x3 * x4;
    };

public:
    void static hash_full_result(Goldilocks::Element (&state)[SPONGE_WIDTH], Goldilocks::Element const (&input)[SPONGE_WIDTH]);
    void static hash(Goldilocks::Element (&state)[CAPACITY], const Goldilocks::Element (&input)[SPONGE_WIDTH]);
};

void PoseidonGoldilocks::hash(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    hash_full_result(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}
void PoseidonGoldilocks::hash_full_result(Goldilocks::Element (&state)[SPONGE_WIDTH], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    std::memcpy(state, input, SPONGE_WIDTH * sizeof(Goldilocks::Element));
    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = state[i] + PoseidonGoldilocksConstants::C[i];
    }

    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            pow7(state[j]);
            state[j] = state[j] + PoseidonGoldilocksConstants::C[(r + 1) * SPONGE_WIDTH + j];
        }

        Goldilocks::Element old_state[SPONGE_WIDTH];
        std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            state[i] = Goldilocks::zero();
            for (int j = 0; j < SPONGE_WIDTH; j++)
            {
                Goldilocks::Element mji = PoseidonGoldilocksConstants::M[j][i];
                mji = mji * old_state[j];
                state[i] = state[i] + mji;
            }
        }
    }

    for (int j = 0; j < SPONGE_WIDTH; j++)
    {
        pow7(state[j]);
        state[j] = state[j] + PoseidonGoldilocksConstants::C[j + (HALF_N_FULL_ROUNDS * SPONGE_WIDTH)];
    }

    Goldilocks::Element old_state[SPONGE_WIDTH];
    std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = Goldilocks::zero();
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            state[i] = state[i] + (PoseidonGoldilocksConstants::P[j][i] * old_state[j]);
        }
    }

    for (int r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        pow7(state[0]);
        state[0] = state[0] + PoseidonGoldilocksConstants::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];

        Goldilocks::Element s0 = state[0] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r];

        for (int j = 1; j < SPONGE_WIDTH; j++)
        {
            s0 = s0 + (state[j] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + j]);
            state[j] = state[j] + (state[0] * PoseidonGoldilocksConstants::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH + j - 1]);
        }
        state[0] = s0;
    }
    for (int r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            pow7(state[j]);
            state[j] = state[j] + PoseidonGoldilocksConstants::C[j + (HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH];
        }

        Goldilocks::Element old_state[SPONGE_WIDTH];
        std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

        for (int i = 0; i < SPONGE_WIDTH; i++)
        {
            state[i] = Goldilocks::zero();
            for (int j = 0; j < SPONGE_WIDTH; j++)
            {
                state[i] = state[i] + (old_state[j] * PoseidonGoldilocksConstants::M[j][i]);
            }
        }
    }

    for (int j = 0; j < SPONGE_WIDTH; j++)
    {
        pow7(state[j]);
    }

    std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);

    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = Goldilocks::zero();
        for (int j = 0; j < SPONGE_WIDTH; j++)
        {
            state[i] = state[i] + (old_state[j] * PoseidonGoldilocksConstants::M[j][i]);
        }
    }
}

#endif