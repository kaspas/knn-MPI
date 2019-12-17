#include <stdlib.h>
#include "knnring.h"
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <stdio.h>

void swap_double(double *a, double *b)
{
    double t = *a;
    *a = *b;
    *b = t;
}
void swap_int(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}
/*
    always take right element as pivot
    so that doesnt need swap 
*/
int partition(int *list,double *arr, int l, int r) 
{ 
    double x = arr[r];
    int i = l; 
    for (int j = l; j <= r - 1; j++) { 
        if (arr[j] <= x) { 
            swap_double(&arr[i], &arr[j]); 
            swap_int(&list[i],&list[j]);
            i++; 
        } 
    } 
    swap_double(&arr[i], &arr[r]); 
    swap_int(&list[i],&list[r]);
    return i; 
} 
  /*
  Quickselect with recursive calls
  k must be valid
  else it returns -1
  */
double kthSmallest(int *list,double *arr, int l, int r, int k) 
{ 

    if (k > 0 && k <= r - l + 1) { 
  
        int index = partition(list,arr, l, r); 

        if (index - l == k - 1) 
            return arr[index]; //it is equal return median value 
  
        if (index - l > k - 1)  
            return kthSmallest(list,arr, l, index - 1, k); //if index greater than k set the boundaries to the left side of the array
  
        return kthSmallest(list,arr, index + 1, r,(k - index + l - 1)); //set the boundaries to the right side and set the k correct
    } 
  
    return -1; 
} 

void quicksort(double * arr,int * idx ,int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(idx,arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quicksort(arr,idx, low, pi - 1);
        quicksort(arr,idx, pi + 1, high);
    }
}
/*
    Variable    Rows    Cols
       X         n       d
       Y         m       d
    Distance     m       n
*/

double * distance(double *X,double *Y,int n,int m,int d){
    double *distance = calloc((long long)(n*m),sizeof(double));
    if(distance==NULL){
        printf("NULL POINTER distance\n");
        exit(1);
    }
    //Calculate -2.0*Y*X' and save in distance
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,n,d,-2.0,Y,d,X,d,0.0,distance,n);
    double *row=(double *)malloc(n*sizeof(double));
    double *col=(double *)malloc(m*sizeof(double));
    if(row==NULL){
        printf("NULL POINTER row\n");
        exit(1);
    }
    if(col==NULL){
        printf("NULL POINTER col\n");
        exit(1);
    }
    int N=(n>=m)?n:m;
    //calculate sum(X.^2) and sum(Y.^2) and save in row and col respectively
    for(long long i=0;i<N;i++){
        double rowtemp=0.0;
        double coltemp=0.0;
        for(long int j=0;j<d;j++){
            if(i<n)rowtemp+=X[i*d+j]*X[i*d+j];
            if(i<m)coltemp+=Y[i*d+j]*Y[i*d+j];
        }
        if(i<n)row[i]=rowtemp;
        if(i<m)col[i]=coltemp;
    }
    //final Euclidean distance calculation based on sqrt( sum(Y.^2)'-2Y*X' + sum(X.^2) )
    for(long long i=0;i<m;i++){
        for(long long j=0;j<n;j++){
            distance[i*n+j]=sqrt(fabs(row[j]+distance[i*n+j]+col[i]));
        }
    }
    free(col);
    free(row);
    return distance;
}
//! Compute k nearest neighbors of each point in X [n-by-d]
/*!

	\param	X	Corpus data points[n-by-d]
	\param	Y	Query data points[m-by-d]
	\param	n	Number of corpus points[scalar]		
	\param	m	Number of query points[scalar]
	\param	d	Number of dimensions[scalar] 
	\param	k	Number of neighbors[scalar]
	
	\return 	The kNN result
*/

knnresult kNN(double* X,double* Y,int n,int m,int d,int k){
    knnresult r;
    r.nidx = (int *)malloc(m*k*sizeof(int));
    r.ndist = (double *)malloc(m*k*sizeof(double));
    r.m=m;
    r.k=k;
    int *index = (int *) malloc(n*sizeof(int));
    double *D = distance(X,Y,n,m,d);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++)index[j]=j;
	//find the k nearest unsorted
        kthSmallest(index,&D[i*n],0,n-1,k);
	//sort them
        quicksort(&D[i*n],index,0,k);
	//copy them to knnresult	
	memcpy(&r.ndist[i*k],&D[i*n],k*sizeof(double));
	memcpy(&r.nidx[i*k],index,k*sizeof(int));
    }
    free(index);
    free(D);
    return r;
}
