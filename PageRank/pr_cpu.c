#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>

int *allocateMemory(int length) {

	int *vec;

	if ((vec = (int *)malloc(length * sizeof(int))) == NULL) {
		fprintf(stderr, "MALLOC ERROR: %s\n", strerror(errno));
		exit(1);
	}
	return vec;
}

// Reads input file into a vector
void *readInput(int nodes, int edges, int* outdegree_count,
		int* indegree_count, int* running_edge_indices, int* edges_1d)
{
	int in,out;
	int edges_read=0;
	int current_in=-1;
	while(edges_read<edges)
	{
		scanf("%d %d",&in,&out);
		indegree_count[in]++;
		outdegree_count[out]++;
		if(in!=current_in)
		{
			running_edge_indices[in]=edges_read;
			current_in=in;
		}
		edges_1d[edges_read]=out;
		edges_read++;
	}
}

int main (int argc, char ** argv) {

	if(argc!=3)
	{
		printf("Invalid Syntax <Nodecount> <Edgecount\n");
		return 0;
	}
	int nodes=atoi(argv[1]);
	int edges=atoi(argv[2]);
	int* outdegree_count=allocateMemory(nodes);
	int* indegree_count=allocateMemory(nodes);
	int* running_edge_indices=allocateMemory(nodes);
	int* edges_1D = allocateMemory(edges);//node1:node2|node2->node1
	readInput(nodes, edges, outdegree_count,indegree_count,
			running_edge_indices, edges_1D);


	free(outdegree_count);
	free(indegree_count);
	free(running_edge_indices);
	free(edges_1D);
	return 0;
}
