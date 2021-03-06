#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <malloc.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKWIDTH 1024

int *allocateMemoryInt(int length) {

   int *vec;

   if ((vec = (int *)malloc(length * sizeof(int))) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }
   memset(vec, 0, length * sizeof(int));
   return vec;
}
double *allocateMemoryDouble(int length) {

   double *vec;

   if ((vec = (double *)malloc(length * sizeof(double))) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }
   return vec;
}

__global__ void CalcPageRank(int nodes, int edges, int *in_d, int *out_d, 
      int *run_d, int *edges_d, double *pagerank_old_d, double *pagerank_new_d) {
   
   int node_index = blockIdx.x * BLOCKWIDTH + threadIdx.x; 

   if(node_index<nodes) {
      double sum = 0;
      double d = 0.85;
      double jumpChance = (1 - d) * (1.0 / nodes);
      int stopIdx = run_d[node_index] + in_d[node_index];
      int k;
      
      for (k = run_d[node_index]; k < stopIdx; k++) {
         int jk = edges_d[k];
         sum += pagerank_old_d[jk] / out_d[jk];
      }  
   
      pagerank_new_d[node_index] = sum * d + jumpChance;
   }
   __syncthreads();
   pagerank_old_d[node_index] = pagerank_new_d[node_index];
}

int main () {

   int i = 0, j = 0, k = 0, run = 0, idx = 0;
   int nodes = 38;
   int edges = 156;
   int iter = 38;
   struct timeval stop, start;

   int* indegree_count=allocateMemoryInt(nodes);
   int* outdegree_count=allocateMemoryInt(nodes);
   int* running_edge_indices=allocateMemoryInt(nodes);
   int* edges_1D = allocateMemoryInt(edges);//node1:node2|node2->node1

   double* pagerank_new;
   double* pagerank_old;

   if ((pagerank_new = (double *)malloc(nodes * sizeof(double))) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }

   if ((pagerank_old = (double *)malloc(nodes * sizeof(double))) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }

   for (i = 0; i < nodes; i++) {
	   pagerank_old[i] = 1.0 / (double)nodes;
   }
   
   memset(pagerank_new, 0, nodes * sizeof(double));
   
   // set starting values for pagerank values to 1/n
   for (i = 0; i < nodes; i++)
      pagerank_old[i] = 1.0 / (double)nodes;
   
   gettimeofday(&start, NULL);
   setvbuf(stdin, NULL, _IOFBF, edges);

   //reads in edges
   for (i = 0; i < edges; i++) {
      scanf("%d\n", &j);
      edges_1D[i] = j;
   }

   //reads in in-degrees, out-degrees, and computes running idx
   for (i = 0; i < nodes; i++) {
      scanf("%d %d %d\n", &idx, &j, &k);
      indegree_count[idx] = j;
      outdegree_count[idx] = k;
      running_edge_indices[idx] = run;
      run += j;
   }

   gettimeofday(&stop, NULL);
   fprintf(stderr, "Read took %lf seconds\n", (stop.tv_sec - start.tv_sec) +
      ((stop.tv_usec - start.tv_usec) / 1000000.0));


   // Begin Cuda Setup 
   int *in_d, *out_d, *run_d, *edges_d;
   double *pagerank_new_d, *pagerank_old_d;
   
   int node_size = nodes * sizeof(int);
   int pr_size = nodes * sizeof(double);
   int edges_size = edges * sizeof(int);
  
   cudaMalloc(&in_d, node_size);
   cudaMemcpy(in_d, indegree_count, node_size, cudaMemcpyHostToDevice);
   
   cudaMalloc(&out_d, node_size);
   cudaMemcpy(out_d, outdegree_count, node_size, cudaMemcpyHostToDevice);
   
   cudaMalloc(&run_d, node_size);
   cudaMemcpy(run_d, running_edge_indices, node_size, cudaMemcpyHostToDevice);
   
   cudaMalloc(&edges_d, edges_size);
   cudaMemcpy(edges_d, edges_1D, edges_size, cudaMemcpyHostToDevice);
   
   cudaMalloc(&pagerank_old_d,pr_size);
   cudaMemcpy(pagerank_old_d, pagerank_old, pr_size, cudaMemcpyHostToDevice);
   
   cudaMalloc(&pagerank_new_d,pr_size);
   //cudaMemcpy(pagerank_new_d, pagerank_new, pr_size, cudaMemcpyHostToDevice);

   int blocks = ceil((double)nodes/(double)BLOCKWIDTH);
   dim3 dimGrid(blocks, 1, 1);
   dim3 dimBlock(BLOCKWIDTH, 1, 1);

   for(i=0; i < iter; i++) {
	    CalcPageRank<<<dimGrid, dimBlock>>>(nodes, edges, in_d, out_d, run_d, edges_d, pagerank_old_d, pagerank_new_d);	    
   }


   cudaMemcpy(pagerank_old, pagerank_old_d, nodes * sizeof(double), cudaMemcpyDeviceToHost);
   gettimeofday(&stop, NULL);
   fprintf(stderr, "Compute took %lf seconds\n", (stop.tv_sec - start.tv_sec) +
      ((stop.tv_usec - start.tv_usec) / 1000000.0));

   cudaFree(in_d);
   cudaFree(out_d);
   cudaFree(run_d);
   cudaFree(edges_d);
   cudaFree(pagerank_old_d);
   cudaFree(pagerank_new_d);

   for (i = 0; i < nodes; i++)
    printf("%.15lf:%d,", pagerank_old[i], i);

   free(indegree_count);
   free(outdegree_count);
   free(pagerank_new);
   free(pagerank_old);
   free(running_edge_indices);
   free(edges_1D);
   return 0;
}






