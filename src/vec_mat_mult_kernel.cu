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
	float Y_temp = 0;

	for (int k = 0; k < MATRIX_SIZE; ++k)
	{
		float A_element = Ad[MATRIX_SIZE * row_number + k]; // Scan through row elements
		float X_element = Xd[k];
		Y_temp += A_element * X_element; 
	}

	Yd[row_number] = Y_temp;
}


__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
