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
__m128 u;

__m256i rnd[GENERATOR_COUNT];
__m128 rndf[GENERATOR_COUNT];
__m128 t;

__m128 x;
__m128 y;
__m128 z;
__m128 uu;
__m128 sq1;


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

// función intrin_sqrt1( )
static inline void intrin_sqrt1()
{
    // Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
    // __m128 _mm_mul_ps (__m128 a, __m128 b)
    __m128 xx_vec = _mm_mul_ps(x, x); // xx_vec = [ x1^2 x2^2 x3^2 x4^2 ]
    __m128 yy_vec = _mm_mul_ps(y, y); // yy_vec = [ y1^2 y2^2 y3^2 y4^2 ]
    __m128 zz_vec = _mm_mul_ps(z, z); // zz_vec = [ z1^2 z2^2 z3^2 z4^2 ]
    
    // Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
    // __m128 _mm_add_ps (__m128 a, __m128 b)
    __m128 partial_1 = _mm_add_ps(xx_vec, yy_vec); // partial_1 = [ (x1^2+y1^2) (x2^2+y2^2) (x3^2+y3^2) (x4^2+y4^2) ]
    __m128 partial_2 = _mm_add_ps(partial_1, zz_vec); // partial_2 = [ (x1^2+y1^2+z1^2) (x2^2+y2^2+z2^2) (x3^2+y3^2+z3^2) (x4^2+y4^2+z4^2) ]

    // Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
    //__m128 _mm_sqrt_ps (__m128 a);
    sq1 = _mm_sqrt_ps(partial_2); // sqrt1_vec = [ sqrt(x1^2+y1^2+z1^2) sqrt(x2^2+y2^2+z2^2) sqrt(x3^2+y3^2+z3^2) sqrt(x4^2+y4^2+z4^2) ]
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

    partial = _mm_div_ps(partial, t);
    // Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
    //__m128 _mm_sqrt_ps (__m128 a);
    uu = _mm_sqrt_ps(partial); // sqrt2_vec = [ sqrt(1.0f-u1^2) sqrt(1.0f-u2^2) sqrt(1.0f-u3^2) sqrt(1.0f-u4^2) ]
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
    x = _mm_set_ps1(0.0f);
    y = _mm_set_ps1(0.0f);
    z = _mm_set_ps1(0.0f);
    u = _mm_set_ps1(0.0f);
    __m128 v = _mm_set_ps1(0.0f);
    __m128 w = _mm_set_ps1(1.0f);
    __m128 weight = _mm_set_ps1(1.0f);

    __m128 one = _mm_set_ps1(1.0f);
    __m128 two = _mm_set_ps1(2.0f);
    __m128 z1  = _mm_set_ps1(0.1f);
    __m128 zz1 = _mm_set_ps1(0.001f);
    __m128i shell_1 = _mm_set1_epi32(SHELLS - 1);

    int mask = 0x0000;

    for (;;) {

        // taylor
        //random_log();
        next(0);
        t = _mm_set_ps(-logf(rndf[0][0]), -logf(rndf[0][1]), -logf(rndf[0][2]), -logf(rndf[0][3]));

        x = _mm_fmadd_ps(t, u, x);
        y = _mm_fmadd_ps(t, v, y);
        z = _mm_fmadd_ps(t, w, z);

        /*
            unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp; // absorb
            if (shell > SHELLS - 1) {
                shell = SHELLS - 1;
            }
        */
        intrin_sqrt1();
        __m128i shell = _mm_set_epi32(sq1[0], sq1[1], sq1[2], sq1[3]);
        __m128i shell_cmp = _mm_cmpgt_epi32(shell, shell_1);
        for(int j=0; j<4; j++) {
            if( shell_cmp[j] == 0xFFFFFFFF ) {
                shell[j] = SHELLS - 1;
            }
        }

        
        __m128 a_w = _mm_mul_ps(albedo, weight);
        __m128 added_heat = _mm_sub_ps(weight, a_w);

        //heat[shell] += added_heat;
        int mask_ = 0x0008;
        for(int j=0; j<4; j++) {
            if ( mask & mask_ == 0x0000 ) {
                heat[shell[j]] += added_heat[j];
            }
            mask_ = mask_ >> 1;
        }
        
        weight = a_w;

        // New direction, rejection method

        __m128 xi1;
        __m128 xi2;
        __m128 gtone;
        int mm = 0;
        
        do {
            next(1);
            next(2);
            
            // xi1 = 2.0f * ((float)rnd[1] / fm) - 1.0f;
            // xi2 = 2.0f * ((float)rnd[2] / fm) - 1.0f;
            // t = xi1 * xi1 + xi2 * xi2;
            xi1 = _mm_fmsub_ps(rndf[1], two, one);
            xi2 = _mm_fmsub_ps(rndf[2], two, one);
            xi1 = _mm_mul_ps(xi1, xi1);
            xi2 = _mm_mul_ps(xi2, xi2);
            __m128 tt = _mm_add_ps(xi1, xi2);
            gtone = _mm_cmp_ps(tt, one, _CMP_LE_OQ);
            mm = _mm_movemask_ps(gtone) | mm;
        
            for(int j=0; j<4; j++) {
                if( tt[j] <= 1.0f ) {
                    t[j] = tt[j];
                }
            }
        } while ( mm != 0x000F );
        //} while (1.0f < t);

        // u = 2.0f * t - 1.0f;
        u = _mm_fmsub_ps(two, t, one);
        
        // float uu = sqrtf(1.0f - u * u);
        intrin_sqrt2();
        v = _mm_mul_ps(xi1, uu); 
        w = _mm_mul_ps(xi2, uu);
     
        int mask1 = _mm_movemask_ps( _mm_cmp_ps(zz1, weight, _CMP_GT_OQ) );
        if ( mask1 != 0x0000 ) {
            next(3);
            int mask2 = _mm_movemask_ps( _mm_cmp_ps(rndf[3], z1, _CMP_GT_OQ) );
            
            int mask3 = 0x0008;
            for(int j=0; j<4; j++) {
                if( mask1 & mask3 ) {
                    if ( mask2 & mask3 ) {
                        mask = mask | mask3;
                    } else {
                        weight[j] = weight[j] * 10.0f;
                    }
                }
                mask3 = mask3 >> 1;
            }

            if ( mask == 0x000F )
                break;
        }
        //if (unlikely( weight < 0.001f )) { // roulette
       //     next(3);
       //     if (((float)rndf[3]) > 0.1f)
       //         break;
       //     weight /= 0.1f;
        
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
    for (unsigned int i = 0; i < PHOTONS / 4; ++i) {
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
    printf("%d\t%lf\t%lf\n", PHOTONS, elapsed, 1e-3 * PHOTONS / elapsed);

    return 0;
}
