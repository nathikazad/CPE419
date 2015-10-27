
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mkl.h>
#include <mkl_vsl.h>
#define DIM 1 /* Task dimension */
#define N 1000 /* Number of observations */
int main()
{
    VSLSSTaskPtr task; /* SS task descriptor */

    double x[N]; /* Array for dataset */
    double mean; /* Array for mean estimate */ 
    double* w = 0; /* Null pointer to array of weights, default weight equal to one will be used in the computation */ 
    MKL_INT p, n, xstorage;
    int status;

    /* Initialize variables used in the computation of mean */
    p = DIM;
    n = N;
    xstorage = VSL_SS_MATRIX_STORAGE_ROWS;
    mean = 0.0;
    
    /* Step 1 - Create task */
    status = vsldSSNewTask( &task, &p, &n, &xstorage, x, w, 0 );

    /* Step 2- Initialize task parameters */
    status = vsldSSEditTask( task, VSL_SS_ED_MEAN, &mean );

    /* Step 3 - Compute the mean estimate using SS one-pass method */
    status = vsldSSCompute(task, VSL_SS_MEAN, VSL_SS_METHOD_1PASS );

    /* Step 4 - deallocate task resources */
    status = vslSSDeleteTask( &task );
    return 0;
}
