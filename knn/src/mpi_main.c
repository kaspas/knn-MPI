#include "knn.h"
#include <stdio.h>
#include <stdlib.h>
#include <cblas-openblas.h>
#include <mpi/mpi.h>
#include <time.h>
#include <sys/time.h>

int main(int argc,char *argv[]){

    if(argc!=4){
        printf("Wrong Usage\n");
        printf("%s n d k , Where\n n = number of points of X array\n",argv[0]);
        printf(" d = number of dimensions\n k = number of nearest points\n");
        exit(1);
    }
    MPI_Init(&argc, &argv);
    int pid,nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    int n=atoi(argv[1])/nproc;
    int d=atoi(argv[2]);
    int k=atoi(argv[3]);
    //printf("%d %d %d\n",n,d,k);
    double *X;
    srand(time(NULL) + pid);
    X=(double*)malloc(n*d*sizeof(double));   
    for (int i=0;i<n;i++){
        for(int j=0;j<d;j++){
            X[i*d+j]=(float)rand()/RAND_MAX * 100;
        }
    }

    knnresult r=distrAllkNN(X,n,d,k);
    free(X);
    free(r.nidx);
    free(r.ndist);
    MPI_Finalize();   
    
    return 0;
}
