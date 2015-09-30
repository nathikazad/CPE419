/*
 * Author: Eric Dazet (edazet) and Nathik Salam (nsalam)
 * CPE 419
 * 2 October 2015
 * 
 * Assignment 2: CUDA Matrix Multiplication
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

#define TILEWIDTH 2

const char* output = "result.out";

#ifdef DOUBLE
typedef double _data_type;
const char* format = "%lf";
const char* printFormat = "%.2lf ";
#else
typedef float _data_type;
const char* format = "%f";
const char* printFormat = "%.2f ";
#endif
