#include "tmwtypes.h"

// Set isevenX = 1 if kernel is even in X, iseven = 0 if odd.
__global__ void superconv4(double *W, const double *A, const double *d,
                           const int32_T Nax, const int32_T Nay, const int32_T Ni, 
                           const int32_T Ndx, const int32_T Ndy, const int32_T Ndxh, 
                           const int32_T Ndyh, const int32_T isevenX, const int32_T isevenY) 
{   
    int32_T x = blockIdx.x; // row of output pixel
    int32_T y = blockIdx.y; // column of output pixel
	int32_T zi = blockIdx.z; // Training example number (4th dimension of A array)
    
    int32_T X = gridDim.x; // W output width in X
    int32_T Y = gridDim.y; // W output height in Y
    //int32_T Zi = gridDim.z; // number of images
    
    int32_T j = threadIdx.x; // Input map number
	int32_T i = threadIdx.y; // Output map number

	int32_T J = blockDim.x; // Number of input maps
	int32_T I = blockDim.y; // Number of input maps

    int32_T mx = x + Ndxh;
    int32_T my = y + Ndyh;
    
    double res = 0;
    int32_T m, n; // x, y of d(1,2)
    #pragma unroll 10
    for (m=-Ndxh; m<=Ndxh-isevenX; m++) {
    	#pragma unroll 10
        for (n=-Ndyh; n<=Ndyh-isevenY; n++) {
            // loop only over 1st and 2nd dimensions
            res += d[Ndx*Ndy*I*zi + Ndx*Ndy*i + (n+Ndyh)*Ndx + (m+Ndxh)] * 
                   A[Nax*Nay*J*zi + Nax*Nay*j + Nax*(my+n) + (mx+m)];
        }        
    }
    
    // (Nm-Nk+1, Nm-Nk+1, Nout, Nin, Ni)
    W[X*Y*J*I*zi + X*Y*J*i + X*Y*j + X*y + x] = res;
}	
