# Compilers
#CC = gcc-10
#CC = clang-11
CC=icc

INC = -I../ -I../../ -I/usr/include/x86_64-linux-gnu
# Flags GCC
#CFLAGS = $(INC) -std=c11 -Wall -Wextra -O3 -march=native -mavx2 -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed

# Flags CLANG
#CFLAGS = $(INC) -std=c11 -Wall -Wextra -O3 -march=native -mavx2 -Rpass=loop-vectorize -Rpass=missed=loopvectorize -Rpass-analysis=loop-vectorize 

# Flags ICC
CFLAGS = $(INC) -std=c11 -Wall -Wextra -O3 -mavx2 -mfma -fopenmp -qopt-report=1 -qopt-report-phase=vec

LDFLAGS = -lm

# Binary file
TARGET = tiny_mc

# Files
C_SOURCES = tiny_mc.c wtime.c
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))

# Rules
all: $(TARGET)

$(TARGET): $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

