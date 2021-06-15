#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "wtime.h"
#include "params.h"

using namespace std;

static void checkCudaCall(cudaError_t statusCode) {
    if(statusCode != cudaSuccess) {
        printf("Error: status code: %d\n", statusCode);
        exit(1);
    }
}

__device__ float next(uint64_t* randomNumber) {

    uint64_t a = 1103515245;
    uint64_t c = 12345;
    uint64_t m = (uint64_t)2<<30;
    float fm = (float)m;

    *randomNumber = *randomNumber * a + c;
    uint64_t s = *randomNumber >> 31;
    s = s << 31;
    *randomNumber = *randomNumber ^ s;
    return (float)*randomNumber / fm;
}

__global__ void photon(uint64_t* rnd, int threadCount) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<threadCount) {

        uint64_t randomNumber = rnd[i];

        const float albedo = MU_S / (MU_S + MU_A);
        const float shells_per_mfp = 1e4 / MICRONS_PER_SHELL / (MU_A + MU_S);
        
        for(int k=0; k<PHOTONS_PER_THREAD; k++) {

            /* launch */
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            float u = 0.0f;
            float v = 0.0f;
            float w = 1.0f;
            float weight = 1.0f;

            for (;;) {
                float t = -logf(next(&randomNumber)); /* move */
                x += t * u;
                y += t * v;
                z += t * w;

                unsigned int shell = sqrtf(x * x + y * y + z * z) * shells_per_mfp; /* absorb */
                if (shell > SHELLS - 1) {
                    shell = SHELLS - 1;
                }
                //heat[shell] += (1.0f - albedo) * weight;
                //heat2[shell] += (1.0f - albedo) * (1.0f - albedo) * weight * weight; /* add up squares */
                weight *= albedo;

                /* New direction, rejection method */
                float xi1, xi2;
                do {
                    xi1 = 2.0f * next(&randomNumber) - 1.0f;
                    xi2 = 2.0f * next(&randomNumber) - 1.0f;
                    t = xi1 * xi1 + xi2 * xi2;
                } while (1.0f < t);
                u = 2.0f * t - 1.0f;
                v = xi1 * sqrtf((1.0f - u * u) / t);
                w = xi2 * sqrtf((1.0f - u * u) / t);

                if (weight < 0.001f) { /* roulette */
                    if (next(&randomNumber) > 0.1f) {
                        // exit
                        break;
                    }
                    weight /= 0.1f;
                }
            }
        }
        rnd[i] = randomNumber;
    }
}

int main() {

    double start = wtime();

    dim3 block(BLOCK_SIZE);
    dim3 grid((PHOTONS + block.x - 1) / block.x / PHOTONS_PER_THREAD);

    uint64_t* rnd = nullptr;

    int threadCount = PHOTONS / PHOTONS_PER_THREAD;
    checkCudaCall( cudaMallocManaged(&rnd, threadCount * sizeof(uint64_t) ) );

    srand(SEED);
    for(int i=0; i<threadCount; i++) {
        rnd[i] = rand()>>2;
    }

    photon<<<grid, block>>>(rnd, threadCount);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaDeviceSynchronize());

    double end = wtime();
    double elapsed = end - start;
    
    cout<<PHOTONS<<"\t"<<elapsed<<"\t"<<(int)(1e-3 * PHOTONS / elapsed)<<endl;

    cudaFree(rnd);

    return 0;
}