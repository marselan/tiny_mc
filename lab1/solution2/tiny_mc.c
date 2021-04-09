/* Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)"
 * 1 W Point Source Heating in Infinite Isotropic Scattering Medium
 * http://omlc.ogi.edu/software/mc/tiny_mc.c
 *
 * Adaptado para CP2014, Nicolas Wolovick
 */

#define _XOPEN_SOURCE 500  // M_PI
#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

/**
 * ******************************************************************* Mersenne Twister
 */
#define UPPER_MASK		0x80000000
#define LOWER_MASK		0x7fffffff
#define TEMPERING_MASK_B	0x9d2c5680
#define TEMPERING_MASK_C	0xefc60000

#include "params.h"
#include "wtime.h"

/**
 * ******************************************************************* Mersenne Twister
 */
#include "mtwister.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * ******************************************************************* Mersenne Twister
 */

inline static void m_seedRand(MTRand* rand, unsigned long seed) {
	rand->mt[0] = seed & 0xffffffff;
	
	for(rand->index=1; rand->index<STATE_VECTOR_LENGTH; rand->index++) {
		rand->mt[rand->index] = (6069 * rand->mt[rand->index-1]) & 0xffffffff;
	} // end for
	
} // end m_seedRand()

/**
* Creates a new random number generator from a given seed.
*/
MTRand seedRand(unsigned long seed) {
	MTRand rand;
	m_seedRand(&rand, seed);
	
	return rand;
} // end seedRand()

/**
 * Generates a pseudo-randomly generated long.
 */
unsigned long genRandLong(MTRand* rand) {

	unsigned long y;
	static unsigned long mag[2] = {0x0, 0x9908b0df}; /* mag[x] = x * 0x9908b0df for x = 0,1 */
	
	if(rand->index >= STATE_VECTOR_LENGTH || rand->index < 0) {
	    /* generate STATE_VECTOR_LENGTH words at a time */
		int kk;
		
		if(rand->index >= STATE_VECTOR_LENGTH+1 || rand->index < 0) {
			m_seedRand(rand, 4357);
		} // end if
		
		for(kk=0; kk<STATE_VECTOR_LENGTH-STATE_VECTOR_M; kk++) {
			y = (rand->mt[kk] & UPPER_MASK) | (rand->mt[kk+1] & LOWER_MASK);
			rand->mt[kk] = rand->mt[kk+STATE_VECTOR_M] ^ (y >> 1) ^ mag[y & 0x1];
		} // end for
		
		for(; kk<STATE_VECTOR_LENGTH-1; kk++) {
			y = (rand->mt[kk] & UPPER_MASK) | (rand->mt[kk+1] & LOWER_MASK);
			rand->mt[kk] = rand->mt[kk+(STATE_VECTOR_M-STATE_VECTOR_LENGTH)] ^ (y >> 1) ^ mag[y & 0x1];
		} // end for
		
		y = (rand->mt[STATE_VECTOR_LENGTH-1] & UPPER_MASK) | (rand->mt[0] & LOWER_MASK);
		rand->mt[STATE_VECTOR_LENGTH-1] = rand->mt[STATE_VECTOR_M-1] ^ (y >> 1) ^ mag[y & 0x1];
		rand->index = 0;
	} // end if

	y = rand->mt[rand->index++];
	y ^= (y >> 11);
	y ^= (y << 7) & TEMPERING_MASK_B;
	y ^= (y << 15) & TEMPERING_MASK_C;
	y ^= (y >> 18);

	return y;
	
} // end genRandLong()

/**
 * Generates a pseudo-randomly generated double in the range [0..1].
 */
double genRand(MTRand* rand) {

	return((double)genRandLong(rand) / (unsigned long)0xffffffff);
	
} // end genRand()

/**
 * ******************************************************************* End Mersenne Twister
 */

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


// global state, heat and heat square in each shell
static float heat[SHELLS];	// arreglo que colecta los lugares donde se va parando el fotón
static float heat2[SHELLS];	// arreglo que eleva al cuadrado el valor anterior para poder compararlos y ver el error


/***
 * Photon
 ***/

static void photon(float arreglo_random[], int cantidad)
{
    int *ptr_nro_random;
    float nro_random;
    const float albedo = MU_S / (MU_S + MU_A);
    const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);

    /* launch */
    float x = 0.0f;		// cartesian coordinate x
    float y = 0.0f;		// cartesian coordinate y
    float z = 0.0f;		// cartesian coordinate z
    float u = 0.0f;		// direction cosine u
    float v = 0.0f;		// direction cosine v
    float w = 1.0f;		// direction cosine w
    float weight = 1.0f;	// photon energy
    
    for (int i = 0;;i++) {
	
		ptr_nro_random = &i;
		
		if (*ptr_nro_random == cantidad - 1){
			i = 0;		
			nro_random = arreglo_random[ i ]; /* move */	// t:= photon packet propagation distance
		}
		else {
			nro_random = arreglo_random[ i ]; /* move */	// t:= photon packet propagation distance
		}
	
		float t = -logf(nro_random); /* move */	// t:= photon packet propagation distance
        x += t * u;		// new cartesian coordinate x
        y += t * v;		// new cartesian coordinate y
        z += t * w;		// new cartesian coordinate z

        unsigned int shell = sqrtf(powf(x,2) + powf(y,2) + powf(z,2)) * shells_per_mfp; /* absorb */
        if (shell > SHELLS - 1) { // recordar que SHELLS es el nivel de discretización
            shell = SHELLS - 1;
        }
        
        float a_w = albedo * weight;		// variable declarada para eliminar redundancia
        float added_heat = weight - a_w;	// variable declarada para eliminar redundancia
        heat[shell] += added_heat;
        weight = a_w;

        /* New direction, rejection method */
        
        float xi1, xi2;
        do {
//            xi1 = (rand() << 1) * d;		// <<1 hace un shift a la izq para multiplicar por 2 más eficientemente
//            xi2 = (rand() << 1) * d;		// <<1 hace un shift a la izq para multiplicar por 2 más eficientemente
			xi1 = nro_random * 2;
			xi1 = nro_random * 2;
            t = powf(xi1,2) + powf(xi2,2);
        } while (1.0f < t);
        float inv_t = 1 / t;		
        u = 2.0f * t - 1.0f;			// variable declarada para transformar / en *
        float uu = sqrtf(1.0f - powf(u,2));	// variable declarada para eliminar redundancia
        v = xi1 * uu * inv_t;
        w = xi2 * uu * inv_t;

        if (unlikely( weight < 0.001f )) { /* roulette */ // acá decimimos si nos quedamos con el foton o lo descartamos
            //if (rand() / (float)RAND_MAX > 0.1f)
            if (nro_random > 0.1f)
                break;
            weight /= 0.1f;
        }
    } // fin del for
}

static void compute_squares() {			// funcion para computar cuadrados en un bucle, disminuir el trabajo de la memoria
    for(int i=0; i<SHELLS; i++) {
        heat2[i] += powf(heat[i],2); /* add up squares */
    }
}


/***
 * Main matter
 ***/

int main(void)
{
    // heading
    printf("# %s\n# %s\n# %s\n", t1, t2, t3);
    printf("# Scattering = %8.3f/cm\n", MU_S);
    printf("# Absorption = %8.3f/cm\n", MU_A);
    printf("# Photons    = %8d\n#\n", PHOTONS);

 
    // start timer
    double start = wtime();
    
/**
 * ******************************************************************* Mersenne Twister
 */
	MTRand r = seedRand(1337);
	int cantidad = PHOTONS;
	float arreglo_random[ cantidad ];

	for(int i=0; i < cantidad;i++) {
	
		arreglo_random[ i ] = genRand(&r);
	}
	
	for (int i = 0; i < PHOTONS; ++i) {
		
		photon(arreglo_random, cantidad);
	}

/**
 * ******************************************************************* End Mersenne Twister
 */

    compute_squares();
    // stop timer
    double end = wtime();
    assert(start <= end);
    double elapsed = end - start;

    printf("# Radius\tHeat\n");
    printf("# [microns]\t[W/cm^3]\tError\n");
    float mm1 = 1/1e12;		// variable declarada para transformar / en *
    float mm2 = 1/3.0f;		// variable declarada para transformar / en *
    float mm3 = 1/PHOTONS;	// variable declarada para transformar / en *
    float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS * mm1;
    float mm4 = 1/t;
    for (unsigned int i = 0; i < SHELLS - 1; ++i) {
        float mm5 = 1/ (powf(i,2) + i + 1.0 * mm2);
	printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
		heat[i] * mm4 * mm5,
		sqrt(heat2[i] - powf(heat[i],2) * mm3) * mm4 * mm5);
    }
    printf("# extra\t%12.5f\n\n", heat[SHELLS - 1] * mm3);
    printf("# %lf seconds\n", elapsed);
    printf("# %lf K photons per second\n", 1e-3 * PHOTONS / elapsed);

    return 0;
}
