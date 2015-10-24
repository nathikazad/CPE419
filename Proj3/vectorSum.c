/*
 * Author: Eric Dazet (edazet) and Nathik Salam (nsalam)
 * CPE 419
 * 27 October 2015
 * 
 * Assignment 4: Vector Sum
 */

#include "vectorSum.h"
#include <math.h>
#include <sys/time.h>

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}

// Function that allocates memory for the vectors
float *allocateMemory(int length) {
   
   float *vec;

   //if ((vec = (float *)_mm_malloc(length * sizeof(float), VEC_ALIGN)) == NULL) {
   if ((vec = (float *)mkl_malloc(length * sizeof(float), VEC_ALIGN)) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }
   return vec;
}

// Reads input file into a vector
float *readFile(char *fileName, int *length) {
   
   int i;
   struct stat st;
   char *iter;
   FILE *inFile;
   float *vec = NULL, val;

   if ((inFile = fopen(fileName, "r")) == NULL) {
      fprintf(stderr, "FILE IO ERROR: %s\n", strerror(errno));
      exit(1);
   }

   if ((i = fstat(fileno(inFile), &st)) < 0) {
      fprintf(stderr, "STAT ERROR! ERROR: %s\n", strerror(errno));
      exit(1);
   }    
   
   if ((iter = (char *)mmap(0, st.st_size, PROT_READ, MAP_PRIVATE,
         fileno(inFile), 0)) == MAP_FAILED) {
      fprintf(stderr, "MMAP ERROR! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   // determines the length of the vector
   for (i = 0; i < st.st_size && iter[i] != '\n'; i++) {
      if (iter[i] == ' ')
         (*length)++;
   }

   if ((i = munmap(iter, st.st_size)) == -1) {
      fprintf(stderr, "MUNMAP ERROR! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   vec = allocateMemory(*length);

   // loads vector
   for (i = 0; i < *length; i++) {
      fscanf(inFile, "%f", &val);
      vec[i] = val;
   }

   fclose(inFile);
   return vec;
}

float *vectorSummation(float *a, float *b, int length) {

   float *restrict c = allocateMemory(length);
   int i = 0;
   int numThreads = omp_get_num_threads();

   #pragma offload target(mic:0) in (i) in(a:length(length)) in(b:length(length)) out(c:length(length))
   {
      #pragma omp parallel for simd private(i) shared(a, b, c)
      for (i = 0; i < length; i++)
         c[i] = a[i] + b[i];
   }

   return c;
}

// writes the product vector to an output file
void outputVector(float *vec, int length) {

   FILE *outFile = fopen(output, "w+");
   int i, j;
   
   for (i = 0; i < length; i++)
      fprintf(outFile, "%.2f ", vec[i]);
   //fprintf(outFile, "\n");
   fclose(outFile);
}

int main(int argc, char **argv) {

   float *restrict a = NULL, *restrict b = NULL, * c;
   int lengthA = 0, lengthB = 0;

   if (argc != 3) {
      fprintf(stderr, "PLEASE SPECIFY FILES! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   // Loads the first input vector
   a = readFile(argv[1], &lengthA);

   // Loads the second input vector
   b = readFile(argv[2], &lengthB);

   fprintf(stderr, "%d == %d\n", lengthA, lengthB);

   // Checks to see if the input vectors can be added together
   if (lengthA != lengthB) {
	   fprintf(stderr, "INVALID MATRICES! ERROR: %d != %d\n", lengthA, lengthB);
	   exit(1);
   }

   // Allocates memory for the product vector
   fprintf(stderr, "Done reading files\n");
   struct timeval tvBegin, tvEnd, tvDiff;
   gettimeofday(&tvBegin, NULL);

   c = vectorSummation(a, b, lengthA);
   gettimeofday(&tvEnd, NULL);
   timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
   printf("%ld.%06ld\n", tvDiff.tv_sec, tvDiff.tv_usec);

   fprintf(stderr, "Writing output\n");
   outputVector(c, lengthA);
   
   // TODO:Calls the histogram function 

   // Frees vector memory
   mkl_free(a);
   mkl_free(b);
   mkl_free(c);

   return 0;
}
