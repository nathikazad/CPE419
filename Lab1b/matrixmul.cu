/*
 * Author: Eric Dazet (edazet) and Nathik Salam (nsalam)
 * CPE 419
 * 2 October 2015
 * 
 * Assignment 2: CUDA Matrix Multiplication
 */

#include "matrixmul.h"

// Function that allocates memory for the arrays
_data_type *allocateMemory(int width) {
   
   _data_type *mat;

   if ((mat = (_data_type *)malloc(width * width * sizeof(_data_type))) == NULL) {
	   fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
	   exit(1);
   }
   return mat;
}

// Reads input file into array
_data_type *readFile(char *fileName, int *width) {
   
   int i;
   struct stat st;
   char *iter;
   FILE *inFile;
   _data_type *mat = NULL, val;

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

   // determines the width of the matrix
   for (i = 0; i < st.st_size && iter[i] != '\n'; i++) {
      if (iter[i] == ' ')
         (*width)++;
   }

   if ((i = munmap(iter, st.st_size)) == -1) {
      fprintf(stderr, "MUNMAP ERROR! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   mat = allocateMemory(*width);

   // TODO: load matrix 
   for (i = 0; i < *width * *width; i++) {
      fscanf(inFile, format, &val);
      mat[i] = val;
   }

   fclose(inFile);
   return mat;
}

// writes the product matrix to an output file
void outputMatrix(_data_type *mat, int width) {

   FILE *outFile = fopen(output, "w+");
   int i, j;
   
   for (i = 0; i < width; i++) {
      for (j = 0; j < width; j++)
         fprintf(outFile, printFormat, mat[i * width + j]);
      fprintf(outFile, "\n");
   }
   fclose(outFile);
}

__global__ void MatMulKernel (_data_type *Md, _data_type *Nd, _data_type *Pd, int width) {
   
   int row = blockIdx.y * TILEWIDTH + threadIdx.y;
   int col = blockIdx.x * TILEWIDTH + threadIdx.x;
   
   float pVal = 0;
   int k;

   for (k = 0; k < width; k++)
      pVal += Md[row * width + k] * Nd[k * width + col];

   printf("Pd[%d] or Pd[%d,%d] = %f \n", row * width + col, row, col, pVal);
   Pd[row * width + col] = pVal;
}

void matrixMulOnDevice (_data_type *m, _data_type *n, _data_type *p, int width) {

   int size = width * width * sizeof(_data_type);
   int blocks = width / TILEWIDTH;
   _data_type *Md, *Nd, *Pd;

   cudaMalloc(&Md, size);
   cudaMemcpy(Md, m, size, cudaMemcpyHostToDevice);
   cudaMalloc(&Nd, size);
   cudaMemcpy(Nd, n, size, cudaMemcpyHostToDevice);
   cudaMalloc(&Pd, size);

   //Launch Kernel
   dim3 dimGrid(blocks, blocks);
   dim3 dimBlock(TILEWIDTH, TILEWIDTH);


   MatMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);
   cudaMemcpy(Pd, p, size, cudaMemcpyDeviceToHost);
   int i;
   for(i=0;i<36;i++)
      printf(" p[%d] = %f\n", i ,p[i]);   
   
   outputMatrix(p, width);

   cudaFree(Md);
   cudaFree(Nd);
   cudaFree(Pd);
}

int main(int argc, char **argv) {

   _data_type *m = NULL, *n = NULL, *p;
   int widthM = 0, widthN = 0;

   if (argc != 3) {
      fprintf(stderr, "PLEASE SPECIFY FILES! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   //loads the first input matrix
   m = readFile(argv[1], &widthM);

   //loads the second input matrix
   n = readFile(argv[2], &widthN);

   // checks to see if the input matrices can be multiplied
   if (widthM != widthN) {
	   fprintf(stderr, "INVALID MATRICES! ERROR: %d != %d\n", widthM, widthN);
	   exit(1);
   }

   // NOTE FROM NATHIK : check height as well?

   //allocates memory for the product matrix
   p = allocateMemory(widthM);
   matrixMulOnDevice(m, n, p, widthM);

   free(m);
   free(n);
   free(p);

   return 0;
}
