SHELL := /bin/bash
CC = gcc-7
MPICC = mpicc
MPIRUN = mpirun -np 4


all: t test_sequential test_mpi test_NB

t:
	cd ..; make lib;
	cp ../lib/*.a ../inc/knn.h ./;
	
test_sequential:
	$(CC) tester.c knn_sequential.a -o $@ -lm -lopenblas
	./test_sequential
	rm test_sequential

test_mpi:
	$(MPICC) tester_mpi.c knn_sync.a knn_sequential.a -o $@ -lm -lopenblas
	$(MPIRUN) ./test_mpi
	rm test_mpi

test_NB:	
	$(MPICC) tester_mpi.c knn_async.a knn_sequential.a -o $@ -lm -lopenblas
	$(MPIRUN) ./test_NB
	rm test_NB
