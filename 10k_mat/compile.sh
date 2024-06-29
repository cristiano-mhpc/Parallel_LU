#!/bin/bash 


#modules
module load boost/1.83.0--openmpi--4.1.6--gcc--12.2.0
module load cuda 

#remove some output files
rm auto Constant.txt Constant1.txt

#host code compilation
mpicxx -c LU.cpp -o LU.o  -lboost_mpi -lboost_thread -lboost_serialization -lboost_system -pthread -lmpfr -lgmp

#device code compilation
nvcc -c kernel.cu -o kernel.o

mpicxx -o auto LU.o kernel.o -L$CUDA_HOME/lib64 -lcudart -lboost_mpi -lboost_serialization -lboost_thread -lboost_system -pthread -lmpfr -lgmp
