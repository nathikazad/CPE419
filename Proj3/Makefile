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

make: vectorSum.o
	$(GCC) $(CFLAGS) -o vSum vectorSum.o

mic: CFLAGS += -mmic
mic: make

vectorSum.o: vectorSum.c
	$(GCC) $(CFLAGS) -c vectorSum.c
	
clean:
	rm vectorSum.o vSum 
	
