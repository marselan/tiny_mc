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
int photons = 0;
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

    __m128 zero = _mm_setzero_ps();
    __m128 one = _mm_set_ps1(1.0f);
    __m128 two = _mm_set_ps1(2.0f);
    __m128 ten = _mm_set_ps1(10.0f);
    __m128 z1  = _mm_set_ps1(0.1f);
    __m128 zz1 = _mm_set_ps1(0.001f);
    __m128 shell_ = _mm_set1_ps((float)SHELLS);
    __m128 shell_1 = _mm_set1_ps((float)(SHELLS - 1));
    __m128 all_true = _mm_cmp_ps(zero, one, _CMP_LT_OQ);

    #if __INTEL_COMPILER
    __m128 _one = _mm_set_ps1(-1.0f);
    #endif

    __m128 process_mask = all_true;
    int bit_mask = 0xFFFF;

    for (;;) {

        //random_log();
        next(0);
        #if __INTEL_COMPILER
        t = _mm_log_ps(rndf[0]);
        t = _mm_mul_ps(t, _one);
        #else
        t = _mm_set_ps(-logf(rndf[0][3]), -logf(rndf[0][2]), -logf(rndf[0][1]), -logf(rndf[0][0]));
        #endif

        x = _mm_fmadd_ps(t, u, x);
        y = _mm_fmadd_ps(t, v, y);
        z = _mm_fmadd_ps(t, w, z);

        
        //    unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp; // absorb
        //    if (shell > SHELLS - 1) {
        //        shell = SHELLS - 1;
        //    }
        
        intrin_sqrt1();
        __m128 shell = _mm_mul_ps(sq1, shells_per_mfp);
        __m128 shell_cmp = _mm_cmp_ps(shell, shell_, _CMP_GE_OQ);
        shell = _mm_blendv_ps(shell, shell_1, shell_cmp);
        __m128i shell_i = _mm_cvtps_epi32(shell);
        uint32_t * shell_index = (uint32_t*)&shell_i;

        
        __m128 a_w = _mm_mul_ps(albedo, weight);
        __m128 added_heat = _mm_sub_ps(weight, a_w);

        //heat[shell] += added_heat;
        
        int mask = 0x0001;
        for(int j=0; j<4; j++) {
            if ( (bit_mask & mask) == mask ) {
                heat[shell_index[j]] += added_heat[j];
            }
            mask = mask << 1;
        }
        
        weight = a_w;

        // New direction, rejection method

        __m128 xi1;
        __m128 xi2;
        __m128 tt;
        __m128 lt_one;
        int mm = 0;
        t = _mm_setzero_ps();
        
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
            tt = _mm_add_ps(xi1, xi2);
            lt_one = _mm_cmp_ps(tt, one, _CMP_LT_OQ);
            mm = _mm_movemask_ps(lt_one) | mm;
            t = _mm_blendv_ps(t, tt, lt_one);

        } while ( mm != 0x000F );
        //} while (1.0f < t);

        // u = 2.0f * t - 1.0f;
        u = _mm_fmsub_ps(two, t, one);
        
        // float uu = sqrtf((1.0f - u * u) / t);
        intrin_sqrt2();
        v = _mm_mul_ps(xi1, uu); 
        w = _mm_mul_ps(xi2, uu);
     
        // e = Energy (weight) low
        // r = Roulette
        __m128 e = _mm_cmp_ps(zz1, weight, _CMP_GT_OQ);
        __m128 _e = _mm_cmp_ps(zz1, weight, _CMP_LE_OQ);

        next(3);
        __m128 r = _mm_cmp_ps(rndf[3], z1, _CMP_GT_OQ);
        __m128 _r = _mm_cmp_ps(rndf[3], z1, _CMP_LE_OQ);
        __m128 e_r = _mm_and_ps(e, _r);
        e_r = _mm_and_ps(process_mask, e_r);
        weight = _mm_blendv_ps(weight, _mm_mul_ps(weight, ten), e_r);

        __m128 F = _mm_or_ps( _mm_and_ps(_e, _r), _mm_and_ps(_e, r) );
        F = _mm_or_ps(F, _mm_and_ps(e, _r));
        process_mask = _mm_and_ps(process_mask, F);
        bit_mask = _mm_movemask_ps(process_mask);  
        if ( bit_mask == 0x0000 ) break;
            
/*
        
        if ( mask1 != 0x0000 ) {

            int bit_count = 0;
            while(mask1) {
                mask1 &= (mask1 - 1);
                bit_count++;
            }
            photons += bit_count;
            break;
*/

            
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
    for (int i = 0; i < PHOTONS / 4; i++) {
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
