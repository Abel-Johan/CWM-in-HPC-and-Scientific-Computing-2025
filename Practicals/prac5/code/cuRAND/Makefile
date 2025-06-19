INC	:= -I$(CUDA_HOME)/include.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcuda -lcurand

GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_80 --ptxas-options=-v -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

EXCFILE = cuRAND_part_c


all: clean compile

compile: Makefile
	$(NVCC) -o $(EXCFILE) cuRAND_part_c.cu $(LIB) $(NVCCFLAGS)

clean:
	rm -f *.o $(ANALYZE)
