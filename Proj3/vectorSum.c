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

   if ((vec = (float *)malloc(length * sizeof(float))) == NULL) {
	   fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
	   exit(1);
   }
   return vec;
}

// Reads input file into a vector
_data_type *readFile(char *fileName, int *length) {
   
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
      fscanf(inFile, format, &val);
      vec[i] = val;
   }

   fclose(inFile);
   return vec;
}

float *vectorSummation(float *a, float *b, int length) {

   float *c = allocateMemory(length);
   int i = 0;

   for (int i = 0; i < length; i++)
      c[i] = a[i] + b[i];

   return c;
}

// writes the product vector to an output file
void outputVector(float *vec, int length) {

   FILE *outFile = fopen(output, "w+");
   int i, j;
   
   for (i = 0; i < length; i++) {
         fprintf(outFile, printFormat, vec[i]);
      fprintf(outFile, "\n");
   }
   fclose(outFile);
}

int main(int argc, char **argv) {

   float *a = NULL, *b = NULL, *c;
   int lengthA = 0, lengthB = 0;

   if (argc != 3) {
      fprintf(stderr, "PLEASE SPECIFY FILES! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   // Loads the first input vector
   m = readFile(argv[1], &lengthA);

   // Loads the second input vector
   n = readFile(argv[2], &lengthB);

   // Checks to see if the input vectors can be added together
   if (lengthA != lengthB) {
	   fprintf(stderr, "INVALID MATRICES! ERROR: %d != %d\n", lengthA, lengthB);
	   exit(1);
   }

   // Allocates memory for the product vector
   c = vectorSummation(a, b, lengthA);
   outputVector(c, length);
   
   // TODO:Calls the histogram function 

   // Frees vector memory
   free(a);
   free(b);
   free(c);

   return 0;
}
