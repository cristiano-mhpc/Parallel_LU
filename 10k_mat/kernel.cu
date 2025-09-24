#include "cuda_runtime.h"
#include "kernel.h"


#ifndef TILE
#define TILE 32
#endif


// Row-major index helper: ld = number of columns 

__device__ __forceinline__ int idx(int r, int c, int ld) {
    return r * ld + c;
}

__global__ void matrixMultKernel(const float* A,const float* B, float* C, int N, int M) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N)  return;

	float acc = 0.0f;

    for (int k = 0; k < M; k++) {
         // C[NxN] = A[NxM] * B[MxN], all row-major
        acc = __fmaf_rn(A[idx(row, k, M)], B[idx(k, col, N)], acc);
    }
	C[idx(row, col, N)] = acc;
}


void matrixMult(const float *A, const float *B, float *C, int N, int M){

    if (N <= 0 || M <= 0) return; // nothing to do

    dim3 block(TILE, TILE, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y, 1);

    matrixMultKernel<<<grid,block>>>(A, B, C, N, M);

}
