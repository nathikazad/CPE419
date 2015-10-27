
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mkl.h>
#define DIM 1 /* Task dimension */
#define N 10 /* Number of observations */
int main()
{
    VSLSSTaskPtr task; /* SS task descriptor */

    double x[N]={1,2,3,4,5,6,7,8,9,10}; /* Array for dataset */
    double mean[DIM],variation[DIM], r2m[DIM], c2m[DIM]; 
    /* mean/variation/2nd raw/2nd central moments */

    double* w = 0;
    MKL_INT p, n, xstorage, covstorage;
    int status;

    /* Initialize variables */
    p = DIM; n = N;
    xstorage = VSL_SS_MATRIX_STORAGE_ROWS;

    /* Create task */ int errcode = vsldSSNewTask( &task, &p, &n, &xstorage, x, w, 0 );

    /* Edit task parameters */
    errcode = vsldSSEditTask( task, VSL_SS_ED_VARIATION, variation );
    errcode = vsldSSEditMoments( task, mean, r2m, 0, 0, c2m, 0, 0 );

    /* Computation of several estimates using 1PASS method */
    MKL_INT estimates = VSL_SS_MEAN| VSL_SS_VARIATION;
    status = vsldSSCompute( task, estimates, VSL_SS_METHOD_1PASS );

    /* De-allocate task resources */
    status = vslSSDeleteTask( &task );
    printf("Mean: %f Standard Deviation:%f\n",mean[0],variation[0]*mean[0]);
    LAPACKE_dlasrt ('d' , N , x );
    int i;
    for(i=0;i<N;i++)
        printf("%f ", x[i]);
     printf("\n");
    return 0;
}
