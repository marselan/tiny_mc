# Compilers
#CC = gcc-10
CC = clang-11

INC = -I../ -I../../
# Flags
#Descomentar para flags_1
#CFLAGS = $(INC) -std=c11 -Wall -Wextra -O1 -march=native -ffast-math -mavx2 -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed

#Descomentar para flags_2
#CFLAGS = $(INC) -std=c11 -Wall -Wextra -O3 -march=native -ffast-math -mavx2 -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed

#Descomentar para flags_3
#CFLAGS = $(INC) -std=c11 -Wall -Wextra -O3 -march=skylake -ffast-math -mavx2 -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed

#Descomentar para flags_4
#CFLAGS = $(INC) -std=c11 -Wall -Wextra -O1 -march=native -ffast-math -mavx2 -Rpass=loop-vectorize -Rpass=missed=loopvectorize -Rpass-analysis=loop-vectorize 

#Descomentar para flags_5
#CFLAGS = $(INC) -std=c11 -Wall -Wextra -O3 -march=native -ffast-math -mavx2 -Rpass=loop-vectorize -Rpass=missed=loopvectorize -Rpass-analysis=loop-vectorize 

#Descomentar para flags_6
CFLAGS = $(INC) -std=c11 -Wall -Wextra -O3 -march=skylake -ffast-math -mavx2 -Rpass=loop-vectorize -Rpass=missed=loopvectorize -Rpass-analysis=loop-vectorize 

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


