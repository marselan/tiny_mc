#include <cstdio>	// remplaza <stdio.h> de C
#include <cstdlib>	// remplaza <stdlib.h> de C
#include <cmath>	// remplaza <math.h> de C
#include <cstdint>	// remplaza <stdint.h> de C
#include <cassert>	// remplaza <assert.h> de C

#include <cuda.h>
#include <omp.h>	// es necesario? hay algo que use OpenMP?
#include <curand_kernel.h>

#include "wtime.h"
#include "params.h"
#include "helper_cuda.h"


__global__ void init_random( curandState *state, uint64_t __restrict__ *seed, uint thread_count)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if ( idx < thread_count )
	{
		curand_init ( seed[idx], idx, 0, &state[idx] );
	}
}

__device__ float next( curandState &State )
{
	return curand_uniform( &State );
}


__global__ void photon(curandState *state, float* __restrict__ heat, uint threadCount)
{
	__shared__ float block_heat[SHELLS];

	uint tid = threadIdx.x;
	uint idx = blockIdx.x * blockDim.x + tid;

	if( tid == 0U )
	{
		for( uint k=0; k < SHELLS; k++ )
		{
			block_heat[ k ] = 0.0f;
		}
	}
	__syncthreads();

	if( idx < threadCount )
	{
		curandState r_state = state [ idx ];
        const float albedo = MU_S / (MU_S + MU_A);
        const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);
        
		for( uint j = 0; j < PHOTONS_PER_THREAD; j++ )
		{
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
				float rnd_1 = next( r_state );
				
				float t = -logf(rnd_1); /* move */
				x += t * u;
				y += t * v;
				z += t * w;

				uint shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp; /* absorb */
				if (shell > SHELLS - 1)
				{
					shell = SHELLS - 1;
				}
	
				atomicAdd(&heat[shell], (1.0f - albedo) * weight);
				weight *= albedo;

				/* New direction, rejection method */
				float xi1, xi2;
				
				do
				{
					float rnd_2 = next( r_state );
					float rnd_3 = next( r_state );
					
					xi1 = 2.0f * rnd_2 - 1.0f;
					xi2 = 2.0f * rnd_3 - 1.0f;
					t = xi1 * xi1 + xi2 * xi2;
				} while (1.0f < t);
			
				u = 2.0f * t - 1.0f;
				v = xi1 * sqrtf((1.0f - u * u) / t);
				w = xi2 * sqrtf((1.0f - u * u) / t);

				if (weight < 0.001f)
				{ /* roulette */
					float rnd_4 = next( r_state );
					
					if (rnd_4 > 0.1f)
					{
						// exit
						break;
					}
				
					weight /= 0.1f;
				}
			}
		}

		__syncthreads();

		if(tid == 0)
		{
			for(int k=0; k<SHELLS; k++)
			{
			atomicAdd(&heat[k], block_heat[k]);
			}
		}
    }
}

__host__ void initHeat(float* __restrict__ heat)
{
	for( uint i = 0; i < SHELLS; i++ )
	{
		heat[ i ] = 0.0f;
	}
}

__host__ void computeHeat2(float* __restrict__ heat, float* __restrict__ heat2)
{
	for( uint i = 0; i < SHELLS; i++)
	{
		heat2[ i ] = heat[ i ] * heat[ i ];
	}
}

template<typename T>
__host__ T div_round_up( T N, T block_dimention, T multiplicity )
{
	uint dimention1 = ( N + block_dimention - 1 ) / block_dimention;
	uint dimention2 = dimention1 / multiplicity;
	return dimention2;
}

int main()
{
    double start = wtime();

	dim3 block( BLOCK_SIZE, 1, 1 );
	dim3 grid( div_round_up( PHOTONS, block.x, PHOTONS_PER_THREAD ), 1, 1 );

	uint64_t *seeds = nullptr;
	curandState *state = nullptr;
	float *heat = nullptr;
	float heat2[SHELLS];
	uint threads_count = PHOTONS / PHOTONS_PER_THREAD;
	
	checkCudaCall( cudaMallocManaged( &state, threads_count * sizeof( curandState ) ) );
	checkCudaCall( cudaMallocManaged( &seeds, threads_count * sizeof( uint64_t ) ) );
	checkCudaCall( cudaMallocManaged( &heat, SHELLS * sizeof( float ) ) );

	srand(SEED);
	for(uint k=0; k<threads_count; k++)
	{
		seeds[k] = rand();
	}

	initHeat(heat);

	init_random<<<grid, block>>>( state, seeds, threads_count );
	checkCudaCall( cudaGetLastError() );
	photon<<<grid, block>>>( state, heat, threads_count);
	checkCudaCall( cudaGetLastError() );
	checkCudaCall( cudaDeviceSynchronize() );

	computeHeat2(heat, heat2);

	double end = wtime();
	assert(start <= end);
	double elapsed = end - start;
    
	//printf( "MACROS -> BLOCK_SIZE = %d, PHOTONS = %d, PHOTONS_PER_THREAD = %d, SHELLS = %d\n", BLOCK_SIZE, PHOTONS, PHOTONS_PER_THREAD, SHELLS );
	//printf("DIMENTIONS -> block( %d, 1, 1 ) grid( %d, 1, 1 )\n", BLOCK_SIZE, div_round_up(PHOTONS, block.x, PHOTONS_PER_THREAD) );
	printf("%d\t%lf\t%lf\n", PHOTONS, elapsed, 1e-3 * PHOTONS / elapsed);
	
	checkCudaCall( cudaFree( state ) );
	checkCudaCall( cudaFree( seeds ) );
	checkCudaCall( cudaFree( heat ) );

    return 0;
}
