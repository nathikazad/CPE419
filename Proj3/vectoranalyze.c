// Function that allocates memory for the vectors
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
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
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

// // writes the product vector to an output file
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
   struct timeval tvBegin, tvEnd, tvMeanSTD, tvSortingTime;
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
   
   gettimeofday(&tvBegin, NULL);
   int status = vsldSSCompute( task, estimates, VSL_SS_METHOD_FAST );
   gettimeofday(&tvEnd, NULL);
   timeval_subtract(&tvMeanSTD, &tvEnd, &tvBegin);

   status = vslSSDeleteTask( &task );
   gettimeofday(&tvBegin, NULL);
   LAPACKE_dlasrt ('I' , length , array );
   gettimeofday(&tvEnd, NULL);
   timeval_subtract(&tvSortingTime, &tvEnd, &tvBegin);
   printf("Minimum value: %lf\n", array[0]);
   printf("Maximum value: %lf\n", array[length-1]);
   printf("Mean: %lf Time Taken:%ld.%06ld\n",mean[0],tvMeanSTD.tv_sec, tvMeanSTD.tv_usec);
   printf("Standard Deviation:%lf\n",variation[0]*mean[0]);
   if(length%2)
      printf("Median: %lf\n",array[length/2]);
   else
      printf("Median: %lf\n",(array[length/2]+array[length/2-1])/(double)2.0);
   printf("Time Taken for Mean and Standard Deviation calculation:%ld.%06lds\n", tvMeanSTD.tv_sec, tvMeanSTD.tv_usec);
   printf("Time Taken for Minimum, Median, Maximum and Sorting:%ld.%06lds\n", tvSortingTime.tv_sec, tvSortingTime.tv_usec);
   outputVector(array, length);
   // struct timeval tvBegin, tvEnd, tvDiff;
   // gettimeofday(&tvBegin, NULL);  
   // gettimeofday(&tvEnd, NULL);
   // timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
   // printf("Phi Time: %ld.%06ld\n", tvDiff.tv_sec, tvDiff.tv_usec);



   
   // Frees vector memory
   _mm_free(array);


   return 0;
}