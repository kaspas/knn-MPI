#include "knn.h"
#include <stdio.h>
#include <stdlib.h>
#include <cblas-openblas.h>
#include <sys/time.h>
#include <string.h>


int main(int argc,char *argv[]){

    if(argc!=5){
        printf("Wrong Usage\n");
        printf("%s n m d k , Where\n n = number of points of X array\n m = number of points of Y array\n",argv[0]);
        printf(" d = number of dimensions\n k = number of nearest points\n");
        exit(1);
    }
    int n=atoi(argv[1]);
    int m=atoi(argv[2]);
    int d=atoi(argv[3]);
    int k=atoi(argv[4]);
    double *X=(double*)malloc(n*d*sizeof(double));
    double *Y=(double*)malloc(m*d*sizeof(double));
    if (X==NULL){
        printf("X null\n");
    }
    if (Y==NULL){
        printf("Y null\n");
    }
    for (int i=0;i<n;i++){
        for(int j=0;j<d;j++){
            X[i*d+j]=(float)rand()/RAND_MAX * 100;
        }
    }
    for (int i=0;i<m;i++){
        for(int j=0;j<d;j++){
            Y[i*d+j]=(float)rand()/RAND_MAX * 100;
        }
    }

    knnresult r=kNN(X,Y,n,m,d,k);
    free(X);
    free(Y);
    free(r.nidx);
    free(r.ndist);
    return 0;
}
