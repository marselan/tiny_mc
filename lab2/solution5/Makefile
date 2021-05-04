# Compilers
CC = gcc-10

INC = -I../ -I../../
# Flags
CFLAGS = $(INC) -std=c11 -Wall -Wextra -O3 -march=native -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed -mavx2
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

