/*
 * Author: Eric Dazet (edazet) and Nathik Salam (nsalam)
 * CPE 419
 * 9 October 2015
 * 
 * Assignment 3: CUDA Matrix Multiplication with Tiling
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
         fprintf(outFile, printFormat, mat[i * numCols + j]);
      fprintf(outFile, "\n");
   }
   fclose(outFile);
}

// GPU function: matrix multiplication per thread
__global__ void MatMulKernel (_data_type *Md, _data_type *Nd, _data_type *Pd, 
   int rowsM, int colsM, int rowsN, int colsN, int iter) {

   __shared__ _data_type Mds[TILEWIDTH][TILEWIDTH];
   __shared__ _data_type Nds[TILEWIDTH][TILEWIDTH];
   
   int row = blockIdx.y * TILEWIDTH + threadIdx.y;
   int col = blockIdx.x * TILEWIDTH + threadIdx.x;
   
   float pVal = 0;
   int k, i;
   
   //possible ternary statements for non-matrix tile values
   for (i = 0; i < (rowsM + TILEWIDTH - 1) / TILEWIDTH; i++) {

      //printf("t.x = %d, t.y = %d, i = %d, b.x = %d, b.y = %d, row = %d, col = %d\n", threadIdx.x, threadIdx.y, i, blockIdx.x, blockIdx.y, row, col);
      Mds[threadIdx.y][threadIdx.x] = ((row >= rowsM) || 
         (TILEWIDTH * i + threadIdx.x >= colsM)) ? 0 : Md[row * colsM + 
         (i * TILEWIDTH + threadIdx.x)];
      //printf("b[%d][%d], Mds[%d][%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.y, threadIdx.x, Mds[threadIdx.y][threadIdx.x]);
      Nds[threadIdx.y][threadIdx.x] = ((col >= colsN) || 
         ((TILEWIDTH * i) + threadIdx.y >= rowsN)) ? 0: Nd[col + 
         (i * TILEWIDTH + threadIdx.y) * colsN];
      //printf("b[%d][%d], Nds[%d][%d] = %f\n", blockIdx.x, blockIdx.y,threadIdx.y, threadIdx.x, Nds[threadIdx.y][threadIdx.x]);
      //Mds[threadIdx.y][threadIdx.x] = Md[row * colsM + (i * TILEWIDTH + threadIdx.x)];
      //Nds[threadIdx.y][threadIdx.x] = Nd[col + (i * TILEWIDTH + threadIdx.y) * colsN];
         
      __syncthreads();
      
      for (k = 0; k < TILEWIDTH; k++)
         pVal += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        
      __syncthreads();
   }
   if (row < rowsM && col < colsN) {
      Pd[row * colsN + col] = pVal;
      //printf("Pd[%d] = %f\n", row * colsN + col, pVal);
   }
}

// GPU setup function
void matrixMulOnDevice (_data_type *m, _data_type *n, _data_type *p, int rowsM, int colsM, 
   int rowsN, int colsN) {

   int sizeM = rowsM * colsM * sizeof(_data_type);  //TILEWIDTH = 32, in header file
   printf("sizeM = %d\n", sizeM);
   int sizeN = rowsN * colsN * sizeof(_data_type);
   printf("sizeN = %d\n", sizeN);
   int sizeP = rowsM * colsN * sizeof(_data_type);
   printf("sizeP = %d\n", sizeP);
   _data_type *Md, *Nd, *Pd;

   // Allocates space and moves matrices to GPU
   if (cudaMalloc(&Md, sizeM) == cudaErrorMemoryAllocation) { 
	printf("CUDA MALLOC ERROR: Md\n");
	exit(1);
   }
   cudaMemcpy(Md, m, sizeM, cudaMemcpyHostToDevice);
   if (cudaMalloc(&Nd, sizeN) == cudaErrorMemoryAllocation) { 
	printf("CUDA MALLOC ERROR: Md\n");
	exit(1);
   }
   cudaMemcpy(Nd, n, sizeN, cudaMemcpyHostToDevice);
   if (cudaMalloc(&Pd, sizeP) == cudaErrorMemoryAllocation) { 
	printf("CUDA MALLOC ERROR: Md\n");
	exit(1);
   }

   // Launch Kernel
   dim3 dimBlock(TILEWIDTH, TILEWIDTH);   //32 x 32 = 1024 threads per block
   dim3 dimGrid((rowsM + dimBlock.x - 1) / dimBlock.x, 
      (colsN + dimBlock.y - 1) / dimBlock.y);          //grid = blocks x blocks 
   
   fprintf(stderr, "Grid.x = %d\n", (rowsM + dimBlock.x - 1) / dimBlock.x);
   fprintf(stderr, "Grid.y = %d\n", (colsN + dimBlock.y - 1) / dimBlock.y);

   int iter = rowsM > colsN ? rowsM : colsN;

   // Calls the matrix multiply function and writes data from GPU to host
   MatMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, rowsM, colsM, rowsN, colsN, iter);
   cudaMemcpy(p, Pd, sizeP, cudaMemcpyDeviceToHost);

   /*for (sizeP = 0; sizeP < rowsM; sizeP++) {
      for (sizeM = 0; sizeM < colsN; sizeM++)
         printf("p[%d] = %f ", sizeP * colsN + sizeM, p[sizeP * colsN + sizeM]);
      printf("\n");
   }*/
   
   outputMatrix(p, rowsM, colsN);

   // Frees GPU memory
   cudaFree(Md);
   cudaFree(Nd);
   cudaFree(Pd);
}

int main(int argc, char **argv) {

   _data_type *m = NULL, *n = NULL, *p = NULL;
   int rowsM = 0, rowsN = 0, colsM = 0, colsN = 0;;

   if (argc != 3) {
      fprintf(stderr, "PLEASE SPECIFY FILES! ERROR: %s\n", strerror(errno));
      exit(1);
   }

   // Loads the first input matrix
   m = readFile(argv[1], &rowsM, &colsM);
   fprintf(stderr, "M: rows, cols = %d, %d\n", rowsM, colsM);

   // Loads the second input matrix
   n = readFile(argv[2], &rowsN, &colsN);
   fprintf(stderr, "N: rows, cols = %d, %d\n", rowsN, colsN);

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
