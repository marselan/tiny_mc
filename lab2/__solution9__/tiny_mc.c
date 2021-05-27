/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */

#define _XOPEN_SOURCE 500 // M_PI

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
__m256 u;

__m256i rnd1[GENERATOR_COUNT];
__m256i rnd2[GENERATOR_COUNT];
__m256 rndf[GENERATOR_COUNT];
__m256 t;

__m256 x;
__m256 y;
__m256 z;
__m256 uu;
__m256 sq1;


static inline void next(int i)
{
    rnd1[i] = _mm256_mul_epi32(rnd1[i], a);
    rnd1[i] = _mm256_add_epi64(rnd1[i], c);
    __m256i s1 = _mm256_srl_epi64(rnd1[i], s);
    __m256i s2 = _mm256_sll_epi64(s1, s);
    rnd1[i] = _mm256_xor_si256(rnd1[i], s2);

    rnd2[i] = _mm256_mul_epi32(rnd2[i], a);
    rnd2[i] = _mm256_add_epi64(rnd2[i], c);
    s1 = _mm256_srl_epi64(rnd2[i], s);
    s2 = _mm256_sll_epi64(s1, s);
    rnd2[i] = _mm256_xor_si256(rnd2[i], s2);

    float m = (float)((uint64_t)2 << 30);
    rndf[i] = _mm256_set_ps((float)rnd1[i][3] / m, (float)rnd1[i][2] / m, (float)rnd1[i][1] / m, (float)rnd1[i][0] / m, (float)rnd2[i][3] / m, (float)rnd2[i][2] / m, (float)rnd2[i][1] / m, (float)rnd2[i][0] / m);
}

// función intrin_sqrt1( )
static inline void intrin_sqrt1()
{
    // Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
    // __m128 _mm_mul_ps (__m128 a, __m128 b)
    __m256 xx_vec = _mm256_mul_ps(x, x); // xx_vec = [ x1^2 x2^2 x3^2 x4^2 ]
    __m256 yy_vec = _mm256_mul_ps(y, y); // yy_vec = [ y1^2 y2^2 y3^2 y4^2 ]
    __m256 zz_vec = _mm256_mul_ps(z, z); // zz_vec = [ z1^2 z2^2 z3^2 z4^2 ]
    
    // Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
    // __m128 _mm_add_ps (__m128 a, __m128 b)
    __m256 partial_1 = _mm256_add_ps(xx_vec, yy_vec); // partial_1 = [ (x1^2+y1^2) (x2^2+y2^2) (x3^2+y3^2) (x4^2+y4^2) ]
    __m256 partial_2 = _mm256_add_ps(partial_1, zz_vec); // partial_2 = [ (x1^2+y1^2+z1^2) (x2^2+y2^2+z2^2) (x3^2+y3^2+z3^2) (x4^2+y4^2+z4^2) ]

    // Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
    //__m128 _mm_sqrt_ps (__m128 a);
    sq1 = _mm256_sqrt_ps(partial_2); // sqrt1_vec = [ sqrt(x1^2+y1^2+z1^2) sqrt(x2^2+y2^2+z2^2) sqrt(x3^2+y3^2+z3^2) sqrt(x4^2+y4^2+z4^2) ]
}

// función intrin_sqrt( )
static inline void intrin_sqrt2()
{
    // Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
    // __m128 _mm_mul_ps (__m128 a, __m128 b)
    __m256 uu_vec = _mm256_mul_ps(u, u); // uu_vec = [ u1^2 u2^2 u3^2 u4^2 ]

    // Broadcast single-precision (32-bit) floating-point value a to all elements of dst.
    // __m128 _mm_set1_ps (float a)
    __m256 id_vec = _mm256_set1_ps(1.0f); // id_vec = [ 1.0f 1.0f 1.0f 1.0f ]

    // Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
    // __m128 _mm_sub_ps (__m128 a, __m128 b)
    __m256 partial = _mm256_sub_ps(id_vec, uu_vec); // partial = [ (1.0f-u1^2) (1.0f-u2^2) (1.0f-u3^2) (1.0f-u4^2) ]

    partial = _mm256_div_ps(partial, t);
    // Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst.
    //__m128 _mm_sqrt_ps (__m128 a);
    uu = _mm256_sqrt_ps(partial); // sqrt2_vec = [ sqrt(1.0f-u1^2) sqrt(1.0f-u2^2) sqrt(1.0f-u3^2) sqrt(1.0f-u4^2) ]
}
/*
 * FIN DE FUNCIÓN INTRINSICS LOGARITMO
 */

/***
 * Photon
 ***/

static void photon(void)
{
    
    __m256 albedo = _mm256_set1_ps(MU_S / (MU_S + MU_A));
    __m256 shells_per_mfp = _mm256_set1_ps(1e4 / MICRONS_PER_SHELL / (MU_A + MU_S));

    
    x = _mm256_set1_ps(0.0f);
    y = _mm256_set1_ps(0.0f);
    z = _mm256_set1_ps(0.0f);
    u = _mm256_set1_ps(0.0f);
    __m256 v = _mm256_set1_ps(0.0f);
    __m256 w = _mm256_set1_ps(1.0f);
    __m256 weight = _mm256_set1_ps(1.0f);

    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 ten = _mm256_set1_ps(10.0f);
    __m256 z1  = _mm256_set1_ps(0.1f);
    __m256 zz1 = _mm256_set1_ps(0.001f);
    __m256 shell_1 = _mm256_set1_ps((float)(SHELLS - 1));
    __m256 all_true = _mm256_cmp_ps(zero, one, _CMP_LT_OQ);

    #if __INTEL_COMPILER
    __m256 _one = _mm256_set1_ps(-1.0f);
    #endif

    __m256 process_mask = all_true;
    int bit_mask = 0x00FF;
    

    for (;;) {

        //random_log();
        next(0);
        #if __INTEL_COMPILER
        t = _mm256_log_ps(rndf[0]);
        t = _mm256_mul_ps(t, _one);
        #else
        t = _mm256_set_ps(-logf(rndf[0][7]), -logf(rndf[0][6]), -logf(rndf[0][5]), -logf(rndf[0][4]), -logf(rndf[0][3]), -logf(rndf[0][2]), -logf(rndf[0][1]), -logf(rndf[0][0]));
        #endif

        x = _mm256_fmadd_ps(t, u, x);
        y = _mm256_fmadd_ps(t, v, y);
        z = _mm256_fmadd_ps(t, w, z);

        //    unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp; // absorb
        //    if (shell > SHELLS - 1) {
        //        shell = SHELLS - 1;
        //    }
        
        intrin_sqrt1();
        __m256 shell = _mm256_mul_ps(sq1, shells_per_mfp);
        __m256 shell_cmp = _mm256_cmp_ps(shell, shell_1, _CMP_GT_OQ);
        shell = _mm256_blendv_ps(shell, shell_1, shell_cmp);
        __m256i shell_i = _mm256_cvtps_epi32(shell);
        uint32_t * shell_index = (uint32_t*)&shell_i;

        
        __m256 a_w = _mm256_mul_ps(albedo, weight);
        __m256 added_heat = _mm256_sub_ps(weight, a_w);
   
        //heat[shell] += added_heat;
        
        int mask = 0x0001;
        for(int j=0; j<8; j++) {
            if ( (bit_mask & mask) == mask ) {
                heat[shell_index[j]] += added_heat[j];
            }
            mask = mask << 1;
        }
        
        weight = a_w;

        // New direction, rejection method

        __m256 xi1;
        __m256 xi2;
        __m256 tt;
        __m256 lt_one;
        int mm = 0;
        t = _mm256_setzero_ps();
        
        do {
            next(1);
            next(2);
            
            // xi1 = 2.0f * ((float)rnd[1] / fm) - 1.0f;
            // xi2 = 2.0f * ((float)rnd[2] / fm) - 1.0f;
            // t = xi1 * xi1 + xi2 * xi2;
            xi1 = _mm256_fmsub_ps(rndf[1], two, one);
            xi2 = _mm256_fmsub_ps(rndf[2], two, one);
            xi1 = _mm256_mul_ps(xi1, xi1);
            xi2 = _mm256_mul_ps(xi2, xi2);
            tt = _mm256_add_ps(xi1, xi2);
            lt_one = _mm256_cmp_ps(tt, one, _CMP_LT_OQ);
            mm = _mm256_movemask_ps(lt_one) | mm;
            t = _mm256_blendv_ps(t, tt, lt_one);

        } while ( mm != 0x00FF );
        //} while (1.0f < t);

        // u = 2.0f * t - 1.0f;
        u = _mm256_fmsub_ps(two, t, one);
        
        // float uu = sqrtf((1.0f - u * u) / t);
        intrin_sqrt2();
        v = _mm256_mul_ps(xi1, uu); 
        w = _mm256_mul_ps(xi2, uu);
     
        // e = Energy (weight) low
        // r = Roulette
        __m256 e = _mm256_cmp_ps(zz1, weight, _CMP_GT_OQ);
        __m256 _e = _mm256_cmp_ps(zz1, weight, _CMP_LE_OQ);

        next(3);
        __m256 r = _mm256_cmp_ps(rndf[3], z1, _CMP_GT_OQ);
        __m256 _r = _mm256_cmp_ps(rndf[3], z1, _CMP_LE_OQ);
        __m256 e_r = _mm256_and_ps(e, _r);
        e_r = _mm256_and_ps(process_mask, e_r);
        weight = _mm256_blendv_ps(weight, _mm256_mul_ps(weight, ten), e_r);

        __m256 F = _mm256_or_ps( _mm256_and_ps(_e, _r), _mm256_and_ps(_e, r) );
        F = _mm256_or_ps(F, _mm256_and_ps(e, _r));
        process_mask = _mm256_and_ps(process_mask, F);
        bit_mask = _mm256_movemask_ps(process_mask);  
        if ( bit_mask == 0x0000 ) break;
            


            
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
        rnd1[g] = _mm256_set_epi32(0, rand() >> 2, 0, rand() >> 2, 0, rand() >> 2, 0, rand() >> 2);
        rnd2[g] = _mm256_set_epi32(0, rand() >> 2, 0, rand() >> 2, 0, rand() >> 2, 0, rand() >> 2);
    }
    // start timer
    double start = wtime();
    // simulation
    for (int i = 0; i < PHOTONS / 8; i++) {
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
