NVFLAGS= -O3 -g -ccbin g++ -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
# list .c and .cu source files here

ICC = icc
CFLAGS = -O2 -openmp -g -restrict -debug inline-debug-info

make: pr_phi
make: pr_cuda

pr_phi: pr_phi.o
	$(ICC) $(CFLAGS) -o pr_phi pr_phi.o
	
pr_phi.o: pr_phi.c
	$(ICC) $(CFLAGS) -c pr_phi.c
	
test: test.o
	$(ICC) $(CFLAGS) -o pr_test test.o

test.o: test.c
	$(ICC) $(CFLAGS) -c test.c

pr_cuda: pr_cuda.cu
	nvcc $(NVFLAGS) -o pr_cuda pr_cuda.cu

prof: cuda_prof.cu
	nvcc $(NVFLAGS) -o cuda_prof cuda_prof.cu

clean: 
	rm -f *.o pr_phi pr_cuda

