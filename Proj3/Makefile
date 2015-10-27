# 
# Eric Dazet (edazet), Nathik Salam (nsalam)
# CPE 419
# 27 October 2015
#
# Project 3 Makefile
#

GCC = icc
CFLAGS = -O3 -openmp -g -restrict -std=c99 -debug inline-debug-info
CFLAGS += -qopt-report=5 -qopt-report-phase=vec -mkl

make: vectorsum.o vectoranalyze.o
	$(GCC) $(CFLAGS) -o vectorsum vectorsum.o
	$(GCC) $(CFLAGS) -mkl -o vectoranalyze mean_example.o

mic: CFLAGS += -mmic -offload-build
mic: make

vectorsum.o: vectorsum.c
	$(GCC) $(CFLAGS) -c vectorsum.c

vectoranalyze.o: mean_example.c
	$(GCC) $(CFLAGS) -c vectoranalyze.c
	
clean:
	rm vectorSum.o vectorsum 
	
