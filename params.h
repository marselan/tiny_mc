#pragma once

#include <time.h> // time

#ifndef SHELLS
#define SHELLS  101// discretization level
#endif

#ifndef PHOTONS
#define PHOTONS 2097152 // photons
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

#ifndef PHOTONS_PER_THREAD
#define PHOTONS_PER_THREAD 8
#endif

#ifndef MU_A
#define MU_A 2.0f // Absorption Coefficient in 1/cm !!non-zero!!
#endif

#ifndef MU_S
#define MU_S 20.0f // Reduced Scattering Coefficient in 1/cm
#endif

#ifndef MICRONS_PER_SHELL
#define MICRONS_PER_SHELL 50 // Thickness of spherical shells in microns
#endif

#ifndef SEED
#define SEED (time(NULL)) // random seed
#endif

