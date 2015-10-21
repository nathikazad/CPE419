# 
# Eric Dazet (edazet), Nathik Salam (nsalam)
# CPE 419
# 27 October 2015
#
# Project 3 Makefile
#

GCC = icc
CFLAGS = -O3 -mmic -vec -openmp -g -debug inline-debug-info

make: vectorSum.o
	$(GCC) $(CFLAGS) -o mm vectorSum.o

vectorSum.o: vectorSum.c
	$(GCC) $(CFLAGS) -c vectorSum.c
	
clean:
	rm vectorSum.o 
	
