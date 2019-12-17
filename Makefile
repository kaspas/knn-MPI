# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-7
MPICC = mpicc
MPIRUN = mpirun -np 4

all: comp test_sequential test_mpi_sync test_mpi_async

comp:
	cd knnring; make lib; cd ..	
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	

test_sequential:
	$(CC) tester.c knnring_sequential.a -o $@ -lm -lopenblas
	./test_sequential

test_mpi_sync:
	$(MPICC) tester_mpi.c knnring_sync.a knnring_sequential.a -o $@ -lm -lopenblas
	$(MPIRUN) ./test_mpi_sync
	
test_mpi_async:
	$(MPICC) tester_mpi.c knnring_async.a knnring_sequential.a -o $@ -lm -lopenblas
	$(MPIRUN) ./test_mpi_async
	
clean:
	rm test_* knnring.h knnring_*
