# Compilers
CC = clang

INC = -I../ -I../../
# Flags
CFLAGS = $(INC) -std=c11 -Wall -Wextra -mcpu=native -O3
ifeq ($(CC), gcc)
	CFLAGS += -march=native
else
	CFLAGS += -mcpu=native
endif

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

