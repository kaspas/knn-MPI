#include <stdlib.h>
#include "knnring.h"
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <stdio.h>
#include "mpi/mpi.h"
#include <sys/time.h>

/*
set Indexes to the corresponding part of the Array
*/
void fixIdx(int *idx,int pid,int n,int k){
    for(int i=0;i<n*k;i++){
        idx[i]=pid*n +idx[i];
    }
}

//! Compute distributed all-kNN of points in X
/*!
	\param	X	Data points[n-by-d]
	\param	n	Number of data points[scalar]
	\param	d	Number of dimensions[scalar]
	\param	k	Number of neighbors[scalar]

	\return	The kNN result
*/

knnresult distrAllkNN(double* X,int n,int d,int k){
    knnresult r,R;
    int pid,nproc;
    int rcv,snd;
    MPI_Status stat;
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&pid);
    rcv=(pid-1+nproc)%nproc;
    snd=(pid+1)%nproc;
    double *Y = (double *)malloc(n*d*sizeof(double));
    memcpy(Y,X,n*d*sizeof(double));
    // compute initial kNN and fix Indexes
    r=kNN(X,Y,n,n,d,k);
    fixIdx(r.nidx,rcv,n,k);
    double *y_old = (double *)malloc(n*d*sizeof(double));
    double *x=(double *)malloc(k*sizeof(double));
    int *idx=(int *)malloc(k*sizeof(int));
    for(int i=1;i<nproc;i++){
	//knnring MPI implementation with synchronous calls make sure that deadlocks won't occurs
        if(pid%2){
            MPI_Send(Y,n*d,MPI_DOUBLE,snd,1,MPI_COMM_WORLD);
            MPI_Recv(Y,n*d,MPI_DOUBLE,rcv,MPI_ANY_TAG,MPI_COMM_WORLD,&stat);
        }
        else{
            memcpy(y_old,Y,n*d*sizeof(double));
            MPI_Recv(Y,n*d,MPI_DOUBLE,rcv,MPI_ANY_TAG,MPI_COMM_WORLD,&stat);
            MPI_Send(y_old,n*d,MPI_DOUBLE,snd,1,MPI_COMM_WORLD);
        }
	//compute kNN and fix Indexes for the incoming part 
        R=kNN(Y,X,n,n,d,k);
        fixIdx(R.nidx,((rcv-i+nproc)%nproc),n,k);
	//merge results to knnresult r 
        for(int l=0;l<n;l++){
            int rr = 0;
            int RR = 0;
            for(int j=0;j<k;j++){
                if(R.ndist[l*k+RR]<r.ndist[l*k+rr]){
                    x[j]=R.ndist[l*k+RR];
                    idx[j]=R.nidx[l*k+RR];
                    RR++;
                }
                else{
                    x[j]=r.ndist[l*k+rr];
                    idx[j]=r.nidx[l*k+rr];
                    rr++;
                }
            }
            memcpy( &r.ndist[l*k], x, k*sizeof(double));
            memcpy( &r.nidx[l*k], idx, k*sizeof(int));
        }
        free(R.nidx);
        free(R.ndist);
    }
    free(idx);
    free(x);
    free(y_old);
    free(Y);
    return r;
}
