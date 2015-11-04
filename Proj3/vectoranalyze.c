/*
 * Author: Eric Dazet (edazet) and Nathik Salam (nsalam)
 * CPE 419
 * 27 October 2015
 *
 * Assignment 4: Vector Analysis
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
#include <i_malloc.h>
#include <mkl.h>
#include <sys/time.h>

#ifdef _MIC_
#define VEC_ALIGN 64
#else
#define VEC_ALIGN 32
#endif


int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) -
       (t1->tv_usec + 1000000 * t1->tv_sec);

    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}

double *allocateMemory(int length) {
   
   double *vec;

   if ((vec = (double *)_mm_malloc(length * sizeof(double), VEC_ALIGN)) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }
   return vec;
}

// Reads input file into a vector
double *readFile(char *fileName, int *length) {
   
   int i;
   struct stat st;
   char *iter;
   FILE *inFile;
   double *vec = NULL, val;

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
      fscanf(inFile, "%lf", &val);
      vec[i] = val;
   }

   fclose(inFile);
   return vec;
}

// writes the product vector to an output file
void outputVector(double *vec, int length) {

   FILE *outFile = fopen("sorted_result.out", "w+");
   int i, j;

   for (i = 0; i < length; i++)
      fprintf(outFile, "%.2lf ", vec[i]);
   fclose(outFile);
}

int main(int argc, char **argv) {

   double *array = NULL;
   int length = 0;
   VSLSSTaskPtr task;
   double mean[1],variation[1], r2m[1], c2m[1]; 
   double* w = 0;
   struct timeval tvBegin, tvEnd, tvMeanSTD, tvSortingTime, tvMin, tvMax, tvMedian;
   MKL_INT p, n, xstorage, covstorage;

   if (argc != 2) {
      fprintf(stderr, "PLEASE SPECIFY FILES! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   // Loads the first input vector
   array = readFile(argv[1], &length);

   p = 1;
   n = length;
   xstorage = VSL_SS_MATRIX_STORAGE_ROWS;
   int errcode = vsldSSNewTask( &task, &p, &n, &xstorage, array, w, 0 );
   errcode = vsldSSEditTask( task, VSL_SS_ED_VARIATION, variation );
   errcode = vsldSSEditMoments( task, mean, r2m, 0, 0, c2m, 0, 0 );
   MKL_INT estimates = VSL_SS_MEAN| VSL_SS_VARIATION;
   
   // finds mean and variation, calculates operation time
   gettimeofday(&tvBegin, NULL);
   int status = vsldSSCompute( task, estimates, VSL_SS_METHOD_FAST );
   gettimeofday(&tvEnd, NULL);
   timeval_subtract(&tvMeanSTD, &tvEnd, &tvBegin);

   status = vslSSDeleteTask( &task );

   // sorts array, calculates operation time
   gettimeofday(&tvBegin, NULL);
   LAPACKE_dlasrt ('I' , length , array );
   gettimeofday(&tvEnd, NULL);
   timeval_subtract(&tvSortingTime, &tvEnd, &tvBegin);

   // finds minimum value, calculates operation time
   gettimeofday(&tvBegin, NULL);
   printf("Minimum value: %lf\n", array[0]);
   gettimeofday(&tvEnd, NULL);
   timeval_subtract(&tvMin, &tvEnd, &tvBegin);

   // finds maximum value, calculates operation time
   gettimeofday(&tvBegin, NULL);
   printf("Maximum value: %lf\n", array[length-1]);
   gettimeofday(&tvEnd, NULL);
   timeval_subtract(&tvMax, &tvEnd, &tvBegin);

   // finds median value, calculates operation time
   gettimeofday(&tvBegin, NULL);
   if (length % 2)
      printf("Median: %lf\n", array[length / 2]);
   else
      printf("Median: %lf\n",(array[length / 2] + array[length / 2 - 1])
         / (double)2.0);

   gettimeofday(&tvEnd, NULL);
   timeval_subtract(&tvMedian, &tvEnd, &tvBegin);

   printf("Mean: %lf \n",mean[0]);
   printf("Standard Deviation:%lf\n",variation[0]*mean[0]);

   printf("Time Taken for Mean and Standard Deviation: %ld.%06lds\n",
      tvMeanSTD.tv_sec, tvMeanSTD.tv_usec);

   printf("Time Taken for Sorting: %ld.%06lds\n", tvSortingTime.tv_sec,
      tvSortingTime.tv_usec);

   printf("Time Taken for Minimum: %ld.%06lds\n", tvMin.tv_sec, tvMin.tv_usec);
   printf("Time Taken for Maximum: %ld.%06lds\n", tvMax.tv_sec, tvMax.tv_usec);
   printf("Time Taken for Median: %ld.%06lds\n", tvMedian.tv_sec, tvMedian.tv_usec);

   // writes sorted array to output file
   outputVector(array, length);

   //frees array
   _mm_free(array);

   return 0;
}
