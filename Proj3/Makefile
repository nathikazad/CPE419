# 
# Eric Dazet (edazet), Nathik Salam (nsalam)
# CPE 419
# 27 October 2015
#
# Project 3 Makefile
#

GCC = icc
CFLAGS = -O2 -openmp -g -restrict -debug inline-debug-info

make: vectorSum.o vectoranalyze.o
	$(GCC) $(CFLAGS) -o vectorsum vectorSum.o
	$(GCC) $(CFLAGS) -mkl -mmic -o vectoranalyze vectoranalyze.o

vectorSum.o: vectorSum.c
	$(GCC) $(CFLAGS) -c vectorSum.c

vectoranalyze.o: vectoranalyze.c
	$(GCC) $(CFLAGS) -mkl -mmic -c vectoranalyze.c
	
clean:
	rm vectorSum.o vectorsum vectoranalyze.o vectoranalyze hist.a hist.b hist.c result.out 
	
