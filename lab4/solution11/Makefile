# Compilers
CC = nvcc

INC = -I../ -I../../
# Flags
CFLAGS = $(INC) -Xcompiler="-Wall -Wextra -fPIE -O2" --ptxas-options=-v
LDFLAGS = 

# Binary file
TARGET = tiny_mc

# Files
C_SOURCES = tiny_mc.cu wtime.c
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))

# Rules
all: $(TARGET)

$(TARGET): $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

