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

#define ALIGN 64

int *allocateMemory(int length) {

   int *vec;

   if ((vec = (int *)memalign(ALIGN, length * sizeof(int))) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }
   memset(vec, 0, length * sizeof(int));
   return vec;
}

// Reads input file into a vector
void *readInput(int nodes, int edges, int*restrict *restrict indegree_count, int*restrict *restrict outdegree_count,
   int*restrict *restrict running_edge_indices, int** edges_1d) {
   int in,out;
   int edges_read=0;
   int current_in=-1;

   while(edges_read < edges) {
      scanf("%d %d",&in,&out);
      (*indegree_count)[in]++;
      (*outdegree_count)[out]++;
      if(in!=current_in)
      {
         (*running_edge_indices)[in]=edges_read;
         current_in=in;
      }
      (*edges_1d)[edges_read]=out;
      edges_read++;
   }
   (*running_edge_indices)[nodes] = edges;
}

int main (int argc, char **argv) {

   if(argc!=4) {
      printf("Invalid Syntax <Nodecount> <Edgecount> <Itercount>\n");
      return 0;
   }
   int i = 0, j = 0, k = 0;
   int nodes=atoi(argv[1]);
   int edges=atoi(argv[2]);
   int iter = atoi(argv[3]);
   struct timeval stop, start;

   double d = 0.85;

   int* restrict indegree_count=allocateMemory(nodes);
   int* restrict outdegree_count=allocateMemory(nodes);
   int* restrict running_edge_indices=allocateMemory(nodes + 1);
   int* edges_1D = allocateMemory(edges);//node1:node2|node2->node1

   gettimeofday(&start, NULL);
   readInput(nodes, edges, &indegree_count, &outdegree_count, &running_edge_indices, &edges_1D);

   #pragma offload_transfer target(mic:0) in(edges_1D[0:edges]) signal(edges_1D)

   double* restrict pagerank_new;
   double* restrict pagerank_old;
   double jumpChance = (1 - d) * (1.0 / nodes);

   if ((pagerank_new = (double *)memalign(ALIGN, nodes * sizeof(double))) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }

   if ((pagerank_old = (double *)memalign(ALIGN, nodes * sizeof(double))) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }

   for (i = 0; i < nodes; i++)
      pagerank_old[i] = 1 / (double)nodes;

   memset(pagerank_new, 0, nodes * sizeof(double));

#pragma offload target(mic:0) in(i, j, k) in(indegree_count:length(nodes)) \
   in(outdegree_count:length(nodes)) in(running_edge_indices:length(nodes)) \
   inout(pagerank_old:length(nodes)) in(pagerank_new:length(nodes)) in(edges_1D:length(edges)) \
   wait(edges_1D) 
   {
   double* restrict temp;
   for (i = 0; i < iter; i++) {
      #pragma omp parallel for schedule(static)
      for (j = 0; j < nodes; j++) {
         double sum = 0;
         int stopIdx = running_edge_indices[j] + indegree_count[j];
         #pragma omp parallel for
         for (k = running_edge_indices[j]; k < stopIdx; k++) {
            int jk = edges_1D[k];
            sum += pagerank_old[jk] / outdegree_count[jk];
         }
         pagerank_new[j] = sum * d + jumpChance;
      }
      temp = pagerank_old;
      pagerank_old = pagerank_new;
      pagerank_new = temp;
   }
   }
   gettimeofday(&stop, NULL);
   fprintf(stderr, "took %lf seconds\n", (stop.tv_sec - start.tv_sec) +
      ((stop.tv_usec - start.tv_usec) / 1000000.0));

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
