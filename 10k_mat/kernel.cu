#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include <stdlib.h>

using namespace std;

__global__ void matrixMultKernel(float* A, float* B, float* C, int N, int M) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    //float tmpSum = 0.0f;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
	float tmpSum = 0;

        for (int i = 0; i < M; i++) {
           // tmpSum += A[ROW * M + i] * B[i * N + COL];
	      tmpSum = __fadd_rn(tmpSum, __fmul_rn(A[ROW * M + i] , B[i * N + COL]));
        }

	C[ROW * N + COL] = tmpSum;
    }
 
}


void matrixMult(float *A, float *B, float *C, int N, int M){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    
    if (N > 32){
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }
    

    matrixMultKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N, M);
}
