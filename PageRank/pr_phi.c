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

int *allocateMemory(int length) {

   int *vec;

   if ((vec = (int *)calloc(length, sizeof(int))) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }
   return vec;
}

// Reads input file into a vector
void *readInput(int nodes, int edges, int** outdegree_count,
   int** running_edge_indices, int** edges_1d) {
   int in,out;
   int edges_read=0;
   int current_in=-1;

   while(edges_read < edges) {
      scanf("%d %d",&in,&out);
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

   double d = 0.85;

   int* outdegree_count=allocateMemory(nodes);
   int* running_edge_indices=allocateMemory(nodes + 1);
   int* edges_1D = allocateMemory(edges);//node1:node2|node2->node1

   double* pagerank_new;
   double* pagerank_old;
   double jumpChance = (1 - d) * (1.0 / nodes);
   double *temp;

   if ((pagerank_new = (double *)calloc(nodes, sizeof(double))) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }

   if ((pagerank_old = (double *)calloc(nodes, sizeof(double))) == NULL) {
      fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
      exit(1);
   }

   //#pragma omp parallel for simd
   for (i = 0; i < nodes; i++)
      pagerank_old[i] = 1 / (double)nodes;

   readInput(nodes, edges, &outdegree_count, &running_edge_indices, &edges_1D);

   for (i = 0; i < iter; i++) {
      for (j = 0; j < nodes; j++) {
         double sum = 0;
         for (k = running_edge_indices[j]; k < running_edge_indices[j + 1]; k++) {
            int jk = edges_1D[k];
            fprintf(stderr, "pagerank_old[%d] : %.13lf\n", jk, pagerank_old[jk]);
            fprintf(stderr, "outdegree_count[%d] : %d\n", jk, outdegree_count[jk]);
            fprintf(stderr, "plus: %.13lf\n", pagerank_old[jk] / outdegree_count[jk]);
            sum += pagerank_old[jk] / outdegree_count[jk];
         }
         pagerank_new[j] = sum * d + jumpChance;
      }
      temp = pagerank_old;
      pagerank_old = pagerank_new;
      pagerank_new = temp;
   }

   for (i = 0; i < nodes; i++)
      printf("%.13lf:%d,", pagerank_old[i], i);

   free(outdegree_count);
   free(pagerank_new);
   free(pagerank_old);
   free(running_edge_indices);
   free(edges_1D);
   return 0;
}
