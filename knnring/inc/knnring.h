#ifndef KNNRING_H
#define KNNRING_H

typedef struct knnresult
{int* nidx;//!< Indices (0-based) of nearest neighbors [m-by-k]
double* ndist;//!< Distance of nearest neighbors[m-by-k]
int m;//!< Number of query points[scalar]
int k;//!< Number of nearest neighbors[scalar]
} knnresult;
double * distance(double *X,double *Y,int n,int m,int d);
//! Compute k nearest neighbors of each point in X [n-by-d]
/*!
    \param X    Corpus data points[n-by-d]
    \param Y    Query data points[m-by-d]
    \param n    Number of corpus points[scalar]
    \param m    Number of query points[scalar]
    \param d    Number of dimensions[scalar]
    \param k    Number of neighbors[scalar]
    \return The kNN result
    */

knnresult kNN(double* X,double* Y,int n,int m,int d,int k);
void quicksort(double * arr,int * idx ,int low, int high);
int compare(const void *a,const void *b);
double * distance(double *X,double *Y,int n,int m,int d);
//! Compute distributed all-kNN of points in X
/*!
    \param X    Data points[n-by-d]
    \param n    Number of data points[scalar]
    \param d    Number of dimensions[scalar]
    \param k    Number of neighbors[scalar]
    \return The kNN result
    */

knnresult distrAllkNN(double* X,int n,int d,int k);


double kthSmallest(int *list,double *arr, int l, int r, int k) ;


#endif
