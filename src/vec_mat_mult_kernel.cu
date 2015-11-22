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


#endif // #ifndef _MATRIXMUL_KERNEL_H_
