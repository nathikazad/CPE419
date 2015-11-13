NVFLAGS= -O3 -g -ccbin g++ -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
# list .c and .cu source files here

ICC = icc
CFLAGS = -O2 -openmp -g -restrict -debug inline-debug-info

make: pr_phi

pr_phi: pr_phi.o
	$(ICC) $(CFLAGS) -o pr_phi pr_phi.o
	
pr_phi.o: pr_phi.c
	$(ICC) $(CFLAGS) -c pr_phi.c

pr_cuda: pr_cuda.cu
	nvcc $(NVFLAGS) -o mm_cuda pr_cuda.cu

clean: 
	rm -f *.o pr_phi pr_cuda pr_native

