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

#include <omp.h> // libreria OpenMP

char t1[] = "Tiny Monte Carlo by Scott Prahl (http://omlc.ogi.edu)";
char t2[] = "1 W Point Source Heating in Infinite Isotropic Scattering Medium";
char t3[] = "CPU version, adapted for PEAGPGPU by Gustavo Castellano"
            " and Nicolas Wolovick";


// global state, heat and heat square in each shell
static float heat[SHELLS];
static float heat2[SHELLS];

// random generator parameters
static const uint64_t a = 1103515245;
static const uint64_t c = 12345;
static const uint64_t m = (uint64_t)2<<30;
static const float fm = (float)m;
#define GENERATOR_COUNT 4

/***
 * Initialization and ask memory for random generator
 ***/

uint64_t * rnd_init; // unsigned int de 64 bits

void init_random_numbers() {
    int thread_count = omp_get_num_threads();
    
    // Se calcula el tamaño necesario en memoria
    int total_random_numbers = GENERATOR_COUNT * thread_count;
    int total_size_needed = total_random_numbers * sizeof(uint64_t);

    // Se marca memoria del tamaño calculado para que cuando la toque se pide memoria efectivamente
    rnd_init = malloc(total_size_needed);

    // Se carga el arreglo con todos los #rnd necesarios
    for(int i=0; i < total_random_numbers; i++) 
    {
        rnd_init[i] = (uint64_t)rand();
    }
}

/***
 * Simulation photon function
 ***/

static void photon(uint64_t* rnd)
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
    
    for (;;) 
    {
        rnd[0] = (a * rnd[0] + c) % m;
        float t = -logf((float)rnd[0] / fm);

        x += t * u;
        y += t * v;
        z += t * w;
        
        // absorb
        unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp;
        if (shell > SHELLS - 1) 
        {
            shell = SHELLS - 1;
        }
        float a_w = albedo * weight;
        float added_heat = weight - a_w;

        // Constructores de sincronización
        // opcion 1 -> #pragma omp atomic ¡Ojo! puede no ser correcto
        // opcion 2 -> #pragma omp critical
        
        heat[shell] += added_heat;
        
        weight = a_w;

        /* New direction, rejection method */
        
        float xi1, xi2;
        do 
        {
            rnd[1] = (a * rnd[1] + c) % m;
            rnd[2] = (a * rnd[2] + c) % m; 
            xi1 = 2.0f * ((float)rnd[1] / fm) - 1.0f;
            xi2 = 2.0f * ((float)rnd[2] / fm) - 1.0f;
            t = xi1 * xi1 + xi2 * xi2;
        } while (1.0f < t);
        
        u = 2.0f * t - 1.0f;
        
        float uu = sqrtf((1.0f - u * u) / t);
        v = xi1 * uu;
        w = xi2 * uu;

        // roulette
        if (unlikely( weight < 0.001f ))
        {
            rnd[3] = (a * rnd[3] + c) % m;
            if (((float)rnd[3] / fm) > 0.1f)
                break;
            weight /= 0.1f;
        }
    }
}

/***
 * heat2[ ] charge function
 ***/

static void compute_squares() 
{
    for(int i=0; i<SHELLS; i++) 
    {
        heat2[i] += heat[i] * heat[i]; /* add up squares */
    }
}


/***
 * Main matter
 ***/

int main(void)
{
    // heading
    /*printf("# %s\n# %s\n# %s\n", t1, t2, t3);
    printf("# Scattering = %8.3f/cm\n", MU_S);
    printf("# Absorption = %8.3f/cm\n", MU_A);
    printf("# Photons    = %8d\n#\n", PHOTONS);*/

    // configure RNG
    srand(SEED);

    //for(int g=0; g<GENERATOR_COUNT; g++) {
    //    rnd[g] = rand();
    //}
    
    // start timer
    double start = wtime();
    
    // simulation
    //#pragma omp parallel num_threads(4) shared(heat)
    //#pragma omp parallel num_threads(4) // start parallel execution
    
    #pragma omp parallel num_threads(28) shared(heat,a,c,m,fm)
    {
        // Constructores
        // opcion 1 -> single
        // opcion 2 -> master // sólo valido para num_threads(2) sino segmentation fault ¿?

        #pragma omp single
        init_random_numbers();

        uint64_t rnd[GENERATOR_COUNT];

        for (int i = 0; i < GENERATOR_COUNT; i++)
        {
            rnd[i] = rnd_init[ GENERATOR_COUNT * omp_get_thread_num() + i ];
        }
        
        // politica de planificación de trabajo para usar en loop for
        int chunksize = PHOTONS>>4; // divido por 2^4
        // opcion 1 -> schedule(static,chunksize)
        // opcion 2 -> schedule(dynamic,chunksize)
        // opcion 3 -> schedule(guided) reduction(+:heat)
        // opcion 4 -> schedule(auto) reduction(+:heat)
        
        // Memoria privada y juntar - contra False sharing
        // reduction(+:heat)
        
        #pragma omp for schedule(static,chunksize) reduction(+:heat)
        for (unsigned int i = 0; i < PHOTONS; ++i)
        {
            photon(rnd);
        }
    } // end parallel execution
   
    compute_squares();
    
    // stop timer
    double end = wtime();
    
    assert(start <= end); // ¿?
    double elapsed = end - start;

    /*printf("\n# Radius\tHeat\n");
    printf("# [microns]\t[W/cm^3]\tError\n");
    float t = 4.0f * M_PI * powf(MICRONS_PER_SHELL, 3.0f) * PHOTONS / 1e12;
    for (unsigned int i = 0; i < SHELLS - 1; ++i) {
        printf("%6.0f\t%12.5f\t%12.5f\n", i * (float)MICRONS_PER_SHELL,
        heat[i] / t / (i * i + i + 1.0 / 3.0),
        sqrt(heat2[i] - heat[i] * heat[i] / PHOTONS) / t / (i * i + i + 1.0f / 3.0f));
    }
    printf("\n# extra\t%12.5f\n", heat[SHELLS - 1] / PHOTONS);
    printf("# %lf seconds\n", elapsed);
    
    printf("\n PHOTONS\tTIME\t   PHOTONS/s\n");*/
    printf("%6.0d\t%12.5lf\t%12.5lf\n", PHOTONS, elapsed, 1e-3 * PHOTONS / elapsed);

    return 0;
}
