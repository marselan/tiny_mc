#include "wtime.h"

#define _POSIX_C_SOURCE 199309L
#include <time.h>

double wtime(void)
{
    struct timespec ts;
    #if __MACH__
        clock_gettime(CLOCK_MONOTONIC, &ts);
    #else
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    #endif

    return 1e-9 * ts.tv_nsec + (double)ts.tv_sec;
}
