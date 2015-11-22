/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	int threadY = threadIdx.y;
	int blockY = blockIdx.y;
	int row_number = blockDim.y * blockY + threadY;
	int elemOffset = MATRIX_SIZE * row_number;
	float Y_temp = 0;

	for (int k = 0; k < MATRIX_SIZE; k++ )
	{
		float A_temp = Ad[elemOffset + k];
		float X_temp = Xd[k];
		Y_temp += A_temp * X_temp;
	}

	Yd[row_number] = Y_temp;
}

__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	__shared__ float Xs[TILE_SIZE_OPTIMIZED];
	__shared__ float As[TILE_SIZE_OPTIMIZED*TILE_SIZE_OPTIMIZED];
	int row_number = (blockDim.y * blockIdx.y + threadIdx.y) * MATRIX_SIZE;
	float temp = 0;
	int i = 0;
	int limit = MATRIX_SIZE/blockDim.x;
	int threadPos = threadIdx.y * blockDim.x + threadIdx.x;
	while ( i < limit )
	{
		__syncthreads();
		if ( threadIdx.y == 0 )
			Xs[threadIdx.x] = Xd[threadIdx.x + (i * blockDim.x)];
		__syncthreads();
		As[threadPos] = Ad[row_number + ((i*blockDim.x) + threadIdx.x)];
		temp += (As[threadPos] * Xs[threadIdx.x]);
		i++;		
	}
	As[threadPos] = temp;
	__syncthreads();
	i = blockDim.x/2;
	while ( i != 0 )
	{
		if ( threadIdx.x < i ) 
			As[threadPos] += As[threadPos + i];
		__syncthreads();
		i /= 2;
	}
	
	if ( threadIdx.x == 0 )
		Yd[blockIdx.y * blockDim.y + threadIdx.y] = As[threadPos];
}
/*
__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	__shared__ float Xs[MATRIX_SIZE];
	int row_number = (blockDim.y * blockIdx.y + threadIdx.y) * MATRIX_SIZE;
	float Y_temp = 0;
	int i = threadIdx.y;
	
	while ( i < MATRIX_SIZE )
	{
		Xs[i] = Xd[i];
		i += TILE_SIZE; 
	}
	__syncthreads();
	
	for (int k = 0; k < MATRIX_SIZE; k++ )
		Y_temp += Ad[row_number + k] * Xs[k];

	Yd[blockDim.y * blockIdx.y + threadIdx.y] = Y_temp;
}
*/

#endif // #ifndef _MATRIXMUL_KERNEL_H_
