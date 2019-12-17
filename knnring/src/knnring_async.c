#include <stdlib.h>
#include <string.h>
#include <mpi/mpi.h>
#include <stdio.h>
#include <cblas.h>
#include <math.h>
#include <sys/time.h>
#include "knnring.h"

/*
set Indexes to the corresponding part of the Array
*/
void fixIdx(int *idx,int pid,int n,int k){
    for(int i=0;i<n*k;i++){
        idx[i]=pid*n+idx[i];
    }
}
/*
	calculating global min and max
	kNNs have been sorted 
	first distance of each row is the min 
	last distance of each row is the max
	compare for each row and keep the local min and max
	in the end use MPI_Reduce with MPI_MIN MPI_MAX to find the global min and max respectively
*/

void mpi_min_max(knnresult r){
    double min_total,max_total;
    double local_min,local_max;
    int k = r.k;
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    local_max=r.ndist[k-1];
    local_min=r.ndist[1];
    for(int i=1;i<r.m;i++){
        if(local_min>r.ndist[i*k+1]) local_min=r.ndist[i*k+1];
        if(local_max<r.ndist[i*k+k-1]) local_max=r.ndist[i*k+k-1];
    }
    MPI_Reduce(&local_min,&min_total,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&local_max,&max_total,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(pid==0)printf("GLOBAL MIN = %f , GLOBAL MAX = %f\n",min_total,max_total);
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
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Request req[2];
    MPI_Status s[2];
    int snd = (pid+1)%nproc;
    int rcv = (pid-1+nproc)%nproc;
    double *Y=(double *)malloc(n*d*sizeof(double));
    memcpy(Y,X,n*d*sizeof(double));
    double *Y_new=(double *)malloc(n*d*sizeof(double));
    double *x=(double *)malloc(k*sizeof(double));
    int *idx=(int *)malloc(k*sizeof(int));
    // inital async communication if Number of processes is greater than 1 
    if(nproc>1){
        MPI_Isend(Y,n*d,MPI_DOUBLE,snd,1,MPI_COMM_WORLD,&req[0]);
        MPI_Irecv(Y_new,n*d,MPI_DOUBLE,rcv,1,MPI_COMM_WORLD,&req[1]); 
    }
    //calculate kNN and fix Indexes
    r=kNN(X,X,n,n,d,k);
    fixIdx(r.nidx,rcv,n,k);
    double *temp = Y ;
    Y=Y_new;
    Y_new=temp;
    // wait call for asynchronous send and receive
    if(nproc>1){
        MPI_Waitall(2,req,s);
    }
   //knnring with asynchronous communications
    for(int i=1;i<nproc;i++){
	//async communication calls
        MPI_Isend(Y,n*d,MPI_DOUBLE,snd,1,MPI_COMM_WORLD,&req[0]);
        MPI_Irecv(Y_new,n*d,MPI_DOUBLE,rcv,1,MPI_COMM_WORLD,&req[1]);
	//meanwhile compute kNN and fix Indexes   
        R=kNN(Y,X,n,n,d,k);
        fixIdx(R.nidx,((rcv-i+nproc)%nproc),n,k);
	// merge results 
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
            memcpy(&r.ndist[l*k], x, k*sizeof(double));
            memcpy(&r.nidx[l*k], idx, k*sizeof(int));
        }
        free(R.nidx);
        free(R.ndist);
	//wait for data receive and trasmit
        MPI_Waitall(2,req,s);
	//swap Y pointers
	//there is no nead for memcpy 
        temp = Y ;
        Y=Y_new;
        Y_new=temp;
    }
    //calculate global min max
    mpi_min_max(r);
    free(Y);
    free(x);
    free(idx);
    free(Y_new);
    return r;
}
