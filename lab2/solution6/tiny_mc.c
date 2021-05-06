/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */

#define _XOPEN_SOURCE 500 // M_PI
#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

#include "params.h"
#include "wtime.h"

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


// global state, heat and heat square in each shell
static float heat[SHELLS];
static float heat2[SHELLS];


#define GENERATOR_COUNT 4
__m256i a;
__m256i c;
__m128i s;
__m128 u; // u = [ u1 u2 u3 u4 ] con ui float de 32 bits o 4 bytes (SSE)

__m256i rnd[GENERATOR_COUNT];
__m128 rndf[GENERATOR_COUNT];
__m128 t;

__m128 sqrt2_vec;

static inline void next(int i)
{
    rnd[i] = _mm256_mul_epi32(rnd[i], a);
    rnd[i] = _mm256_add_epi64(rnd[i], c);
    __m256i s1 = _mm256_srl_epi64(rnd[i], s);
    __m256i s2 = _mm256_sll_epi64(s1, s);
    rnd[i] = _mm256_xor_si256(rnd[i], s2);
    float m = (float)((uint64_t)2 << 30);
    rndf[i] = _mm_set_ps((float)rnd[i][3] / m, (float)rnd[i][2] / m, (float)rnd[i][1] / m, (float)rnd[i][0] / m);
}

/*
 * INICIO DE FUNCIÓN INTRINSICS LOGARITMO
 */
void random_log()
{
    __m256 coef = _mm256_set_ps(1.0f, -1 / 2.0f, 1 / 3.0f, -1 / 4.0f, 1 / 5.0f, -1 / 6.0f, 1 / 7.0f, -1 / 8.0f);
    float wlog[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    next(0);
    for (int j = 0; j < 4; j++) {
        float wnum_1 = rndf[0][j] - 1;
        float wnum_2 = wnum_1 * wnum_1;
        float wnum_3 = wnum_2 * wnum_1;
        float wnum_4 = wnum_2 * wnum_2;
        float wnum_5 = wnum_3 * wnum_2;
        float wnum_6 = wnum_3 * wnum_3;
        float wnum_7 = wnum_4 * wnum_3;
        float wnum_8 = wnum_4 * wnum_4;

        __m256 wvec = _mm256_set_ps(wnum_1, wnum_2, wnum_3, wnum_4, wnum_5, wnum_6, wnum_7, wnum_8);
        wvec = _mm256_mul_ps(coef, wvec);

        for (unsigned int i = 0; i < 8; i++) {
            wlog[j] += wvec[i];
        }
    }

    t = _mm_set_ps(-wlog[0], -wlog[1], -wlog[2], -wlog[3]);
}

// función intrin_sqrt( )
static inline void intrin_sqrt2()
{
    // Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
    // __m128 _mm_mul_ps (__m128 a, __m128 b)
    __m128 uu_vec = _mm_mul_ps(u, u); // uu_vec = [ u1^2 u2^2 u3^2 u4^2 ]

    // Broadcast single-precision (32-bit) floating-point value a to all elements of dst.
    // __m128 _mm_set1_ps (float a)
    __m128 id_vec = _mm_set1_ps(1.0f); // id_vec = [ 1.0f 1.0f 1.0f 1.0f ]

    // Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
    // __m128 _mm_sub_ps (__m128 a, __m128 b)
    __m128 partial = _mm_sub_ps(id_vec, uu_vec); // partial = [ (1.0f-u1^2) (1.0f-u2^2) (1.0f-u3^2) (1.0f-u4^2) ]

    // Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
    //__m128 _mm_sqrt_ps (__m128 a);
    sqrt2_vec = _mm_sqrt_ps(partial); // sqrt2_vec = [ sqrt(1.0f-u1^2) sqrt(1.0f-u2^2) sqrt(1.0f-u3^2) sqrt(1.0f-u4^2) ]
}
/*
 * FIN DE FUNCIÓN INTRINSICS LOGARITMO
 */

/***
 * Photon
 ***/

static void photon(void)
{
    __m128 albedo = _mm_set_ps1(MU_S / (MU_S + MU_A));
    __m128 shells_per_mfp = _mm_set_ps1(1e4 / MICRONS_PER_SHELL / (MU_A + MU_S));

    /* launch */
    __m128 x = _mm_set_ps1(0.0f);
    __m128 y = _mm_set_ps1(0.0f);
    __m128 z = _mm_set_ps1(0.0f);
    u = _mm_set_ps1(0.0f);
    __m128 v = _mm_set_ps1(0.0f);
    __m128 w = _mm_set_ps1(1.0f);
    __m128 weight = _mm_set_ps1(1.0f);

    __m128 one = _mm_set_ps1(1.0f);
    __m128 two = _mm_set_ps1(2.0f);

    for (;;) {

        // taylor
        random_log();

        x = _mm_fmadd_ps(t, u, x);
        y = _mm_fmadd_ps(t, v, y);
        z = _mm_fmadd_ps(t, w, z);
        /*
        unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp; // absorb
        if (shell > SHELLS - 1) {
            shell = SHELLS - 1;
        }
        */
        __m128 a_w = _mm_mul_ps(albedo, weight);
        __m128 added_heat = _mm_sub_ps(weight, a_w);
        //heat[shell] += added_heat;
        weight = a_w;

        // New direction, rejection method

        __m128 xi1;
        __m128 xi2;
        __m128 gtone;
        int mm = 0;
        do {
            next(1);
            next(2);
            xi1 = _mm_fmsub_ps(rndf[1], two, one);
            xi2 = _mm_fmsub_ps(rndf[2], two, one);
            xi1 = _mm_mul_ps(xi1, xi1);
            xi2 = _mm_mul_ps(xi2, xi2);
            t = _mm_add_ps(xi1, xi2);
            gtone = _mm_cmp_ps(one, t, _CMP_GE_OQ);
            mm = _mm_movemask_ps(gtone) | mm;
        } while (mm != 0x000F);
        //} while (1.0f < t);
        /*
        float inv_t = 1 / t;
        u = 2.0f * t - 1.0f;
        float uu = sqrtf(1.0f - u * u);
        v = xi1 * uu * inv_t;
        w = xi2 * uu * inv_t;

        if (unlikely( weight < 0.001f )) { // roulette
            next(3);
            if (((float)rndf[3][0]) > 0.1f)
                break;
            weight /= 0.1f;
        }
        */
        break;
    }
}

static void compute_squares()
{
    for (int i = 0; i < SHELLS; i++) {
        heat2[i] += heat[i] * heat[i]; /* add up squares */
    }
}


/***
 * Main matter
 ***/

int main(void)
{
    // heading
    //printf("# %s\n# %s\n# %s\n", t1, t2, t3);
    //printf("# Scattering = %8.3f/cm\n", MU_S);
    //printf("# Absorption = %8.3f/cm\n", MU_A);
    //printf("# Photons    = %8d\n#\n", PHOTONS);

    // configure RNG
    srand(SEED);
    a = _mm256_set1_epi32(1103515245);
    c = _mm256_set1_epi64x(12345);
    s = _mm_set1_epi64x(31);

    for (int g = 0; g < GENERATOR_COUNT; g++) {
        rnd[g] = _mm256_set_epi32(0, rand() >> 2, 0, rand() >> 2, 0, rand() >> 2, 0, rand() >> 2);
    }
    // start timer
    double start = wtime();
    // simulation
    for (unsigned int i = 0; i < PHOTONS; ++i) {
        photon();
    }
    compute_squares();
    // stop timer
    double end = wtime();
    assert(start <= end);
    double elapsed = end - start;
    /*
    printf("# Radius\tHeat\n");
    printf("# [microns]\t[W/cm^3]\tError\n");
    float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
    
    for (unsigned int i = 0; i < SHELLS - 1; ++i) {
        printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
               heat[i] / t / (i * i + i + 1.0 / 3.0),
               sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("# extra\t%12.5f\n\n", heat[SHELLS - 1] / PHOTONS);
    printf("# %lf seconds\n", elapsed);
*/
    printf("%d\t%lf\n", PHOTONS, 1e-3 * PHOTONS / elapsed);

    return 0;
}
