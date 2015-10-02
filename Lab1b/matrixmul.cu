/*
 * Author: Eric Dazet (edazet) and Nathik Salam (nsalam)
 * CPE 419
 * 2 October 2015
 * 
 * Assignment 2: CUDA Matrix Multiplication
 */

#include "matrixmul.h"

// Function that allocates memory for the matrices
_data_type *allocateMemory(int width) {
   
   _data_type *mat;

   if ((mat = (_data_type *)malloc(width * width * sizeof(_data_type))) == NULL) {
	   fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
	   exit(1);
   }
   return mat;
}

// Reads input file into a matrix
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

   // loads matrix
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

// GPU function: matrix multiplication per thread
__global__ void MatMulKernel (_data_type *Md, _data_type *Nd, _data_type *Pd, int width) {
   
   int row = blockIdx.y * TILEWIDTH + threadIdx.y;
   int col = blockIdx.x * TILEWIDTH + threadIdx.x;
   
   float pVal = 0;
   int k;

   for (k = 0; k < width; k++)
      pVal += Md[row * width + k] * Nd[k * width + col];
   Pd[row * width + col] = pVal;
}

// GPU setup function
void matrixMulOnDevice (_data_type *m, _data_type *n, _data_type *p, int width) {

   int size = width * width * sizeof(_data_type);  //TILEWIDTH = 32, in header file
   int blocks = width / TILEWIDTH;                 //determines number of blocks in one dimension
   _data_type *Md, *Nd, *Pd;

   // Allocates space and moves matrices to GPU
   cudaMalloc(&Md, size);
   cudaMemcpy(Md, m, size, cudaMemcpyHostToDevice);
   cudaMalloc(&Nd, size);
   cudaMemcpy(Nd, n, size, cudaMemcpyHostToDevice);
   cudaMalloc(&Pd, size);

   // Launch Kernel
   dim3 dimGrid(blocks, blocks);          //grid = blocks x blocks 
   dim3 dimBlock(TILEWIDTH, TILEWIDTH);   //32 x 32 = 1024 threads per block

   // Calls the matrix multiply function and writes data from GPU to host
   MatMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);
   cudaMemcpy(p, Pd, size, cudaMemcpyDeviceToHost);
   
   outputMatrix(p, width);

   // Frees GPU memory
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

   // Loads the first input matrix
   m = readFile(argv[1], &widthM);

   // Loads the second input matrix
   n = readFile(argv[2], &widthN);

   // Checks to see if the input matrices can be multiplied
   if (widthM != widthN) {
	   fprintf(stderr, "INVALID MATRICES! ERROR: %d != %d\n", widthM, widthN);
	   exit(1);
   }

   // NOTE FROM NATHIK : check height as well?
   // NO, SINCE THE MATRICES WERE KNOWN TO BE SQUARE

   // Allocates memory for the product matrix
   p = allocateMemory(widthM);
   
   // Calls the GPU prep function
   matrixMulOnDevice(m, n, p, widthM);

   // Frees matrix memory
   free(m);
   free(n);
   free(p);

   return 0;
}
