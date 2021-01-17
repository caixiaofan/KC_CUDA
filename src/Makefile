MACRO=-DDEBUG_DF
CXX=g++
CXXFLAGS=-O3 -std=c++14 -fopenmp -pthread $(MACRO)
NVCC=nvcc
NVCCFLAGS=--gpu-architecture=compute_80 --gpu-code=sm_80 -Xcompiler -fopenmp -O3 -std=c++14 $(MACRO)
CC=gcc
CFLAGS+=	-Wall -O2 -g -std=c99

all : kc_cpu kc_gpu

kc_cpu : kc_cpu.o
	$(CXX) $(CXXFLAGS) -o kc_cpu kc_cpu.o

kc_gpu : kc_gpu.o kc_cu.o gpu.o
	$(CXX) $(CXXFLAGS) -o kc_gpu kc_gpu.o kc_cu.o gpu.o -L /usr/local/cuda/lib64 -lcudart

kc_cpu.o : kc_cpu.cpp
	$(CXX) $(CXXFLAGS) -c kc_cpu.cpp

kc_gpu.o : kc_gpu.cpp
	$(CXX) $(CXXFLAGS) -c kc_gpu.cpp

kc_cu.o : kc.cu
	$(NVCC) $(NVCCFLAGS) -dc kc.cu -o kc_cu.o

gpu.o : kc_cu.o
	$(NVCC) $(NVCCFLAGS) -dlink kc_cu.o -o gpu.o

.PHONY : clean
clean :
	rm kc_cpu kc_gpu *.o
