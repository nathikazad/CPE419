/*
 * Author: Eric Dazet (edazet) and Nathik Salam (nsalam)
 * CPE 419
 * 27 October 2015
 * 
 * Assignment 4: Vector Sum
 */

#include "vectorsum.h"
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

   if ((vec = (float *)_mm_malloc(length * sizeof(float), VEC_ALIGN)) == NULL) {
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

float *vectorSummation(float * restrict a, float * restrict b, int length) {

   float *restrict c = allocateMemory(length);
   int i;

   #pragma offload target(mic:0) in (i) in(a:length(length)) in(b:length(length)) out(c:length(length))
   {
      #pragma omp parallel for private(i) shared(a, b, c)
      for (i = 0; i < length; i++)
         c[i] = a[i] + b[i];
   }

   return c;
}

void vectorHistogration(float * restrict vector, int vector_length, int * restrict histogram, int hist_length, float start, float end)
{
   int i, index;
   for(i=0;i<vector_length;i++)
   {
      index=(int)((vector[i]+end)*2);
      if(index==hist_length)
         index--;
      histogram[index] = histogram[index] + 1;
   }
}
// writes the product vector to an output file
void outputVector(float *vec, int length) {

   FILE *outFile = fopen(output, "w+");
   int i, j;

   for (i = 0; i < length; i++)
      fprintf(outFile, "%.2f ", vec[i]);
   fclose(outFile);
}

void outputHistogram(const char* filename, int *hist, int length) {

   FILE *outFile = fopen(filename, "w+");
   int i, j;
   for (i = 0; i < length; i++)
      fprintf(outFile, "%d, %d\n", i, hist[i]);

   fclose(outFile);
}

int main(int argc, char **argv) {

   float *restrict a = NULL, *restrict b = NULL, * c;
   int a_histogram[40] = {0}, b_histogram[40] = {0}, c_histogram[80] = {0};
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
      fprintf(stderr, "INVALID VECTORS! ERROR: %d != %d\n", lengthA, lengthB);
      exit(1);
   }

   // Allocates memory for the product vector
   fprintf(stderr, "Done reading files\n");
   struct timeval tvBegin, tvEnd, tvDiff;
   gettimeofday(&tvBegin, NULL);

   c = vectorSummation(a, b, lengthA);
   gettimeofday(&tvEnd, NULL);
   timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
   printf("Phi Time: %ld.%06ld\n", tvDiff.tv_sec, tvDiff.tv_usec);

   fprintf(stderr, "Writing output\n");
   outputVector(c, lengthA);
  
   // DID ARRAY INIT IN DECLARATION, BELOW LOOP NO LONGER NEEDED
 
   /*int i;
   for(i = 0; i < 80;i++)
   {
      if(i < 40)
      {
         a_histogram[i]=0;
         b_histogram[i]=0;
         c_histogram[i]=0;
      }
      else
         c_histogram[i]=0;
   }*/

   // HISTOGRAM GENERATION IS FINE, 100-MILLION ELEMENT INPUT FILES ARE INCORRECT
   // PIAZZA SAID THE HISTOGRAM FOR THE RESULT VECTOR WILL HAVE ALSO HAVE 40 BINS BUT 
   // LARGER RANGES, MIGHT NEED TO ALTER FUNCTION

   vectorHistogration(a, lengthA, a_histogram, 40, -10.0, 10.0);
   vectorHistogration(b, lengthB, b_histogram, 40, -10.0, 10.0);
   vectorHistogration(c, lengthB, c_histogram, 80, -20.0, 20.0);
 
   outputHistogram(a_histogram_filename,a_histogram, 40);
   outputHistogram(b_histogram_filename,b_histogram, 40);
   outputHistogram(c_histogram_filename,c_histogram, 80);
   
   // Frees vector memory
   _mm_free(a);
   _mm_free(b);
   _mm_free(c);

   return 0;
}
