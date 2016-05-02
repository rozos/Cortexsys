#include "tmwtypes.h"

// Set isevenX = 1 if kernel is even in X, iseven = 0 if odd.
__global__ void superconv5(float *dout, const float *d, const float *W,
                           const int32_T Nax, const int32_T Nay, const int32_T Ni, 
                           const int32_T Ndx, const int32_T Ndy, const int32_T Ndxh, 
                           const int32_T Ndyh, const int32_T isevenX, const int32_T isevenY) 
{   
    int32_T x = blockIdx.x; // row of output pixel
    int32_T y = blockIdx.y; // column of output pixel
	int32_T zi = blockIdx.z; // Training example number (4th dimension of d array)
    
    int32_T X = gridDim.x; // dout output width in X
    int32_T Y = gridDim.y; // dout output height in Y
    int32_T Zi = gridDim.z; // number of images
    
    int32_T j = threadIdx.x; // Input map number
	int32_T i = threadIdx.y; // Output map number

	int32_T J = blockDim.x; // Number of input maps
	int32_T I = blockDim.y; // Number of input maps

	// center point of kernel starts on boundary of d
	int32_T mx = x - Ndxh + isevenX;
    int32_T my = y - Ndyh + isevenY;
    
	// For a "full" convolution, add if statements to set m,n range to simulate zero padded input
    float res = 0;
	float dpad = 0;
    int32_T m, n; // x, y of W(1,2)
    #pragma unroll 10
    for (m=-Ndxh; m<=Ndxh-isevenX; m++) {
    	#pragma unroll 10
        for (n=-Ndyh; n<=Ndyh-isevenY; n++) {

			// Perform full convolution (set d to zero if kernel point is outside of image)
			if (((mx + m) < 0) || ((my + n) < 0) || ((mx + m) >= Nax) || ((my + n) >= Nay)) {
				dpad = 0;
			}
			else {
				dpad = d[Nax*Nay*I*zi + Nax*Nay*i + Nax*(my + n) + (mx + m)];
			}
            // loop only over 1st and 2nd dimensions
			res += W[Ndx*Ndy*J*i + Ndx*Ndy*j + (n + Ndyh)*Ndx + (m + Ndxh)] * dpad;
        }        
    }
    
    // (Nm-Nk+1, Nm-Nk+1, Nout, Nin, Ni)
    dout[X*Y*J*Zi*i + X*Y*J*zi + X*Y*j + X*y + x] = res;
}	
