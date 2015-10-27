/*
 * Author: Eric Dazet (edazet) and Nathik Salam (nsalam)
 * CPE 419
 * 27 October 2015
 * 
 * Assignment 4: Vector Sum
 */

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mkl.h>
#include <i_malloc.h>

const char* output = "result.out";
const char* a_histogram_filename = "hist.a";
const char* b_histogram_filename = "hist.b";
const char* c_histogram_filename = "hist.c";

#ifdef _MIC_
#define VEC_ALIGN 64
#else
#define VEC_ALIGN 32
#endif

