/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */

#define _XOPEN_SOURCE 500  // M_PI
#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

#include "params.h"
#include "wtime.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>

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
__m256i rnd[GENERATOR_COUNT];
__m256d rndf[GENERATOR_COUNT];
  
 static inline void next(int i) {
     rnd[i] = _mm256_mul_epi32(rnd[i], a);
     rnd[i] = _mm256_add_epi64(rnd[i], c);
     __m256i s1 = _mm256_srl_epi64(rnd[i], s);
     __m256i s2 = _mm256_sll_epi64(s1, s);
     rnd[i] = _mm256_xor_si256(rnd[i], s2);
     double m = (double)((uint64_t)2<<30);
     rndf[i] = _mm256_set_pd((double)rnd[i][3]/m, (double)rnd[i][2]/m, (double)rnd[i][1]/m, (double)rnd[i][0]/m);
 }

/***
 * Photon
 ***/

static void photon(void)
{
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

    /* launch */
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float w = 1.0f;
    float weight = 1.0f;

    for (;;) {
        next(0);
        
        // taylor
        float xx[8];
        xx[0] = (rndf[0][0])-1;
        for(int xxi=1; xxi<8; xxi++) xx[xxi] = xx[xxi-1] * xx[0];
        float t = xx[0] - xx[1]/2 + xx[2]/3 - xx[3]/4 + xx[4]/5 - xx[5]/6 + xx[6]/7 - xx[7]/8;
     
        x += t * u;
        y += t * v;
        z += t * w;

        unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp; /* absorb */
        if (shell > SHELLS - 1) {
            shell = SHELLS - 1;
        }
        float a_w = albedo * weight;
        float added_heat = weight - a_w;
        heat[shell] += added_heat;
        weight = a_w;

        /* New direction, rejection method */
        
        float xi1, xi2;
        do {
            next(1);
            next(2);
            xi1 = 2.0f * ((float)rndf[1][0]) - 1.0f;
            xi2 = 2.0f * ((float)rndf[2][0]) - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);
        float inv_t = 1 / t;
        u = 2.0f * t - 1.0f;
        float uu = sqrtf(1.0f - u * u);
        v = xi1 * uu * inv_t;
        w = xi2 * uu * inv_t;

        if (unlikely( weight < 0.001f )) { /* roulette */
            next(3);
            if (((float)rndf[3][0]) > 0.1f)
                break;
            weight /= 0.1f;
        }
    }
}

static void compute_squares() {
    for(int i=0; i<SHELLS; i++) {
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

    for(int g=0; g<GENERATOR_COUNT; g++) {
        rnd[g] = _mm256_set_epi32(0, rand()>>2, 0, rand()>>2, 0, rand()>>2, 0, rand()>>2);
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
