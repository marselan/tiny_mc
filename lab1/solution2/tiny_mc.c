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

static void photon(void)
{
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

    float neg_inv_rand_max = -1.0f / (float)RAND_MAX;	// variable declarada para transformar / en *
    float d = 1 / ((float)RAND_MAX - 1.0f);			// variable declarada para transformar / en *
    for (;;) {
        float t = -logf(rand() * neg_inv_rand_max ); /* move */	// t:= photon packet propagation distance
        x += t * u;		// new cartesian coordinate x
        y += t * v;		// new cartesian coordinate y
        z += t * w;		// new cartesian coordinate z

        unsigned int shell = sqrtf(powf(x,2) + powf(y,2) + powf(z,2)) * shells_per_mfp; /* absorb */
        if (shell > SHELLS - 1) {
            shell = SHELLS - 1;
        }
        float a_w = albedo * weight;		// variable declarada para eliminar redundancia
        float added_heat = weight - a_w;	// variable declarada para eliminar redundancia
        heat[shell] += added_heat;
        weight = a_w;

        /* New direction, rejection method */
        
        float xi1, xi2;
        do {
            xi1 = (rand() << 1) * d;		// <<1 hace un shift a la izq para multiplicar por 2 más eficientemente
            xi2 = (rand() << 1) * d;		// <<1 hace un shift a la izq para multiplicar por 2 más eficientemente
            t = powf(xi1,2) + powf(xi2,2);
        } while (1.0f < t);
        float inv_t = 1 / t;		
        u = 2.0f * t - 1.0f;			// variable declarada para transformar / en *
        float uu = sqrtf(1.0f - powf(u,2));	// variable declarada para eliminar redundancia
        v = xi1 * uu * inv_t;
        w = xi2 * uu * inv_t;

        if (unlikely( weight < 0.001f )) { /* roulette */ // acá decimimos si nos quedamos con el foton o lo descartamos
            if (rand() / (float)RAND_MAX > 0.1f)
                break;
            weight /= 0.1f;
        }
    }
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

    // configure RNG
    srand(SEED);
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
