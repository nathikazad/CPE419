/*
 * Author: Eric Dazet (edazet) and Nathik Salam (nsalam)
 * CPE 419
 * 27 October 2015
 * 
 * Assignment 4: Vector Sum
 */

#include "vectorSum.h"

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
      #pragma omp parallel for private(i) shared(a, b, c, length)
      for (i = 0; i < length; i++)
         c[i] = a[i] + b[i];
   }

   return c;
}

void vectorHistogration(float * restrict vector, int vector_length, int * restrict histogram, 
   int hist_length, float start, float end) {
   
   int* restrict temp;
   int threadCount = omp_get_max_threads();
   int size = threadCount * hist_length * sizeof(int);

   if ((temp = (int *)_mm_malloc(size, VEC_ALIGN)) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }

   memset(temp, 0, size);

   #pragma offload target(mic:0) in (threadCount, vector_length, hist_length, end) in(vector:length(vector_length)) \
      in(temp:length(size)) out(histogram:length(hist_length)) 
   {
      int i, index;
      
      #pragma omp parallel private (i, index) shared (vector, histogram, temp, vector_length, hist_length, end, threadCount)
      {
         int threadId = omp_get_thread_num();
	 #pragma omp for  
	 for (i = 0; i < vector_length; i++) {
	    index = (int)((vector[i] + end) * (hist_length / (2 * end)));
	    if (index == hist_length)
	       index--;
	    temp[hist_length * threadId + index] += 1;
	 }

      }
	
	 //#pragma omp for
	 for (i = 0; i < hist_length; i++) {
	    for (index = 0; index < threadCount; index++)
               histogram[i] += temp[hist_length * index + i];
	}
   }
   _mm_free(temp);
}
// writes the product vector to an output file
void outputVector(float *vec, int length) {

   FILE *outFile = fopen(output, "w+");
   int i;

   for (i = 0; i < length; i++)
      fprintf(outFile, "%.2f ", vec[i]);
   fclose(outFile);
}

void outputHistogram(const char* filename, int *hist, int length) {

   FILE *outFile = fopen(filename, "w+");
   int i;

   for (i = 0; i < length; i++)
      fprintf(outFile, "%d, %d\n", i, hist[i]);
   fclose(outFile);
}

int main(int argc, char **argv) {

   float *restrict a = NULL, *restrict b = NULL, * c;
   int * restrict a_histogram, * restrict b_histogram, * restrict c_histogram;
   int lengthA = 0, lengthB = 0;

   if (argc != 3) {
      fprintf(stderr, "PLEASE SPECIFY FILES! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   omp_set_num_threads(224);

   // Loads the first input vector
   a = readFile(argv[1], &lengthA);

   // Loads the second input vector
   b = readFile(argv[2], &lengthB);

   // Checks to see if the input vectors can be added together
   if (lengthA != lengthB) {
      fprintf(stderr, "INVALID VECTORS! ERROR: %d != %d\n", lengthA, lengthB);
      exit(1);
   }

   c = vectorSummation(a, b, lengthA);

   outputVector(c, lengthA);

   if ((a_histogram = (int *)_mm_malloc(40 * sizeof(int), VEC_ALIGN)) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }
   if ((b_histogram = (int *)_mm_malloc(40 * sizeof(int), VEC_ALIGN)) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }
   if ((c_histogram = (int *)_mm_malloc(40 * sizeof(int), VEC_ALIGN)) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }
 
   vectorHistogration(a, lengthA, a_histogram, 40, -10.0, 10.0);
   vectorHistogration(b, lengthB, b_histogram, 40, -10.0, 10.0);
   vectorHistogration(c, lengthB, c_histogram, 40, -20.0, 20.0);
 
   outputHistogram(a_histogram_filename, a_histogram, 40);
   outputHistogram(b_histogram_filename, b_histogram, 40);
   outputHistogram(c_histogram_filename, c_histogram, 40);
   
   // Frees vector memory
   _mm_free(a);
   _mm_free(b);
   _mm_free(c);

   return 0;
}
