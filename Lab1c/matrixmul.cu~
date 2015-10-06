/*
 * Author: Eric Dazet (edazet) and Nathik Salam (nsalam)
 * CPE 419
 * 2 October 2015
 * 
 * Assignment 2: CUDA Matrix Multiplication
 */

#include "matrixmul.h"

// Function that allocates memory for the matrices
_data_type *allocateMemory(int rows, int cols) {
   
   _data_type *mat;

   if ((mat = (_data_type *)malloc(rows * cols * sizeof(_data_type))) == NULL) {
	   fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
	   exit(1);
   }
   return mat;
}

// Reads input file into a matrix
_data_type *readFile(char *fileName, int *rows, int *cols) {
   
   int i, rCount = 0, cCount = 0;
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
   for (i = 0; i < st.st_size; i++) {
      if (!rCount && iter[i] == ' ')
         cCount++;
      else if (iter[i] == '\n')
         rCount++;
   }

   if ((i = munmap(iter, st.st_size)) == -1) {
      fprintf(stderr, "MUNMAP ERROR! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   mat = allocateMemory(rCount, cCount);

   // loads matrix
   for (i = 0; i < rCount * cCount; i++) {
      fscanf(inFile, format, &val);
      mat[i] = val;
   }

   *rows = rCount;
   *cols = cCount;

   fclose(inFile);
   return mat;
}

// writes the product matrix to an output file
void outputMatrix(_data_type *mat, int numRows, int numCols) {

   FILE *outFile = fopen(output, "w+");
   int i, j;
   
   for (i = 0; i < numRows; i++) {
      for (j = 0; j < numCols; j++)
         fprintf(outFile, printFormat, mat[i * numRows + numCols]);
      fprintf(outFile, "\n");
   }
   fclose(outFile);
}

// GPU function: matrix multiplication per thread
__global__ void MatMulKernel (_data_type *Md, _data_type *Nd, _data_type *Pd, int numRows, numCols) {

   //TODO: Shared memory
   __shared__ _data_type Mds[TILEWIDTH][TILEWIDTH];
   __shared__ _data_type Nds[TILEWIDTH][TILEWIDTH];
   
   int row = blockIdx.y * TILEWIDTH + threadIdx.y;
   int col = blockIdx.x * TILEWIDTH + threadIdx.x;
   
   float pVal = 0;
   int k;

   //TODO: Change to handle non-square matrices
   for (k = 0; k < width; k++)
      pVal += Md[row * numRows + k] * Nd[k * numCols + col];
   Pd[row * width + col] = pVal;
}

// GPU setup function
void matrixMulOnDevice (_data_type *m, _data_type *n, _data_type *p, int rowsM, int colsM, 
   int rowsN, int colsN) {

   int sizeM = rowsM * colsM * sizeof(_data_type);  //TILEWIDTH = 32, in header file
   int sizeN = rowsN * colsN * sizeof(_data_type);
   int sizeP = rowsM * colsN * sizeof(_data_type);
   _data_type *Md, *Nd, *Pd;

   // Allocates space and moves matrices to GPU
   cudaMalloc(&Md, sizeM);
   cudaMemcpy(Md, m, sizeM, cudaMemcpyHostToDevice);
   cudaMalloc(&Nd, sizeN);
   cudaMemcpy(Nd, n, sizeN, cudaMemcpyHostToDevice);
   cudaMalloc(&Pd, sizeP);

   // Launch Kernel
   dim3 dimGrid(//TODO, TODO);          //grid = blocks x blocks 
   dim3 dimBlock(TILEWIDTH, TILEWIDTH);   //32 x 32 = 1024 threads per block

   // Calls the matrix multiply function and writes data from GPU to host
   MatMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, rowsM, colsN);
   cudaMemcpy(p, Pd, sizeP, cudaMemcpyDeviceToHost);
   
   outputMatrix(p, rowsM, colsN);

   // Frees GPU memory
   cudaFree(Md);
   cudaFree(Nd);
   cudaFree(Pd);
}

int main(int argc, char **argv) {

   _data_type *m = NULL, *n = NULL, *p;
   int rowsM = 0, rowsN = 0, colsM = 0, colsN = 0;;

   if (argc != 3) {
      fprintf(stderr, "PLEASE SPECIFY FILES! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   // Loads the first input matrix
   m = readFile(argv[1], &rowsM, &colsM);

   // Loads the second input matrix
   n = readFile(argv[2], &rowsN, &colsN);

   // Checks to see if the input matrices can be multiplied
   if (colsM != rowsN) {
	   fprintf(stderr, "INVALID MATRICES! ERROR: %d != %d\n", colsM, rowsN);
	   exit(1);
   }

   // Allocates memory for the product matrix
   p = allocateMemory(rowsM, colsN);
   
   // Calls the GPU prep function
   matrixMulOnDevice(m, n, p, rowsM, colsM, rowsN, colsN);

   // Frees matrix memory
   free(m);
   free(n);
   free(p);

   return 0;
}
