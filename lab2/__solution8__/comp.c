#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    
    __m128 zero = _mm_setzero_ps();
    __m128 one = _mm_set1_ps(1.0f);
    __m128 _one = _mm_set1_ps(-1.0f);
    __m128 z5 = _mm_set1_ps(-0.5f); 

    __m128 m = _mm_set_ps(2.0f, 0.5f, 1.0f, -1.0f);
    __m128 m2 = _mm_set_ps(2.0f, 0.5f, 0.9f, -1.0f);
    __m128 c;
    __m128 c2;
    __m128 mask_1 = one;
    __m128 mask_2;

    printf("m = %f %f %f %f\n", m[3], m[2], m[1], m[0]);
   printf("m2 = %f %f %f %f\n", m2[3], m2[2], m2[1], m2[0]);
    int mm = 0;
    
        
    c = _mm_cmp_ps(m, one, _CMP_LT_OQ);
    printf("c = %f %f %f %f\n", c[3], c[2], c[1], c[0]);

    c2 = _mm_cmp_ps(m2, one, _CMP_LT_OQ);

    printf("c2 = %f %f %f %f\n", c2[3], c2[2], c2[1], c2[0]);

    c2 = _mm_andnot_ps(c, c2);
    printf("!c & c2 = %f %f %f %f\n", c2[3], c2[2], c2[1], c2[0]);
    /*mm = _mm_movemask_ps(c) | mm;
    mask_1 = _mm_blendv_ps(zero, one, c);

    m = _mm_fmadd_ps(z5, mask_1, m);
    
    printf("mask_1 = %f %f %f %f\n", mask_1[3], mask_1[2], mask_1[1], mask_1[0]);
    printf("m = %f %f %f %f\n", m[3], m[2], m[1], m[0]);
        if (mm == 0x000F) break;
*/
        
        
        
    

    

    return 0;
}