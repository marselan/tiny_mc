#compilador
CU=nvcc

#para incluir el único archivo params.h
INC = -I../ -I../../

#modificación de parametros
CUMACROS= -DBLOCK_SIZE=32U -DPHOTONS=32768U -DPHOTONS_PER_THREAD=8U -DSHELLS=505U

#flags_1
#CUFLAGS= $(INC) -Xcompiler="-Wall -Wextra -fPIE" --ptxas-options=-v $(CUMACROS)

#flags_2
#CUFLAGS= $(INC) -Xcompiler="-Wall -Wextra" --ptxas-options=-v $(CUMACROS)

#flags_3
CUFLAGS= $(INC) -Xcompiler="-Wall -Wextra -O3 -use_fast_math" --prec-sqrt=true --extra-device-vectorization --gpu-architecture=sm_75 --ptxas-options=-v $(CUMACROS)

#flags de librerias
LDFLAGS=-lgomp -lcurand

TARGET=tiny_mc

CU_SOURCES=tiny_mc.cu wtime.c
CU_OBJS = $(patsubst %.c, %.o, $(CU_SOURCES))

all: $(TARGET)

$(TARGET): $(CU_OBJS)
	$(CU) $(CUFLAGS) -o $@ $^ $(LDFLAGS)

# no hay reglas implicitas en nvcc
.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
