#include "tmwtypes.h"

// Note: on GTX Titan Black, optimal number of threads/images is 64 (Ni = 64).
// Set isevenX = 1 if kernel is even in X, iseven = 0 if odd.
__global__ void superconv2(float *Mout, const float *M, const float *K, 
                           const int32_T Nmx,  const int32_T Nmy,
                           const int32_T Nkx,  const int32_T Nky, 
                           const int32_T Nkxh, const int32_T Nkyh, 
                           const int32_T isevenX, const int32_T isevenY) 
{   
    int32_T x = blockIdx.x; // row of output pixel
    int32_T y = blockIdx.y; // column of output pixel
    
    int32_T X = gridDim.x; // map output width in X
    int32_T Y = gridDim.y; // map output height in Y
    
    //int32_T Zk = blockDim.x; // number of 2D kernels (kernel depth)
    int32_T zk = threadIdx.x; // map number (3rd dimension of M array)

    int32_T mx = x + Nkxh;
    int32_T my = y + Nkyh;
    
    float res = 0;
    int32_T i, j;
    #pragma unroll 10
    for (i=-Nkxh; i<=Nkxh-isevenX; i++) {
    	#pragma unroll 10
        for (j=-Nkyh; j<=Nkyh-isevenY; j++) {
            // loop only over 1st and 2nd dimensions
            res += K[Nkx*Nky*zk + (j+Nkyh)*Nkx + (i+Nkxh)] * 
                   M[Nmx*Nmy*zk + Nmx*(my+j) + (mx+i)];
        }        
    }
    
    // (Nm-Nk+1, Nm-Nk+1, Nkz, Ni)
    Mout[X*Y*zk + X*y + x] = res;
}	
