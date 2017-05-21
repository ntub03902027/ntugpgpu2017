#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}


__global__ void CaculateFixed(const float *background, const float *target, const float *mask, float *fixed
	, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			fixed[curt*3+0] = 4 * target[curt*3+0];
			fixed[curt*3+1] = 4 * target[curt*3+1];
			fixed[curt*3+2] = 4 * target[curt*3+2];

			//n_t
			fixed[curt*3+0] -= (yt == 0)? target[curt*3+0] : target[(curt-wt)*3+0];
			fixed[curt*3+1] -= (yt == 0)? target[curt*3+1] : target[(curt-wt)*3+1];
			fixed[curt*3+2] -= (yt == 0)? target[curt*3+2] : target[(curt-wt)*3+2];

			//w_t
			fixed[curt*3+0] -= (xt == 0)? target[curt*3+0] : target[(curt-1)*3+0];
			fixed[curt*3+1] -= (xt == 0)? target[curt*3+1] : target[(curt-1)*3+1];
			fixed[curt*3+2] -= (xt == 0)? target[curt*3+2] : target[(curt-1)*3+2];

			//s_t
			fixed[curt*3+0] -= (yt == ht-1)? target[curt*3+0] : target[(curt+wt)*3+0];
			fixed[curt*3+1] -= (yt == ht-1)? target[curt*3+1] : target[(curt+wt)*3+1];
			fixed[curt*3+2] -= (yt == ht-1)? target[curt*3+2] : target[(curt+wt)*3+2];

			//e_t
			fixed[curt*3+0] -= (xt == wt-1)? target[curt*3+0] : target[(curt+1)*3+0];
			fixed[curt*3+1] -= (xt == wt-1)? target[curt*3+1] : target[(curt+1)*3+1];
			fixed[curt*3+2] -= (xt == wt-1)? target[curt*3+2] : target[(curt+1)*3+2];


			//n_b
			fixed[curt*3+0] += ( (yb == 0) || (yt != 0 && mask[(curt-wt)] > 127.0f) )? 0.0 : background[(curb-wb)*3+0];
			fixed[curt*3+1] += ( (yb == 0) || (yt != 0 && mask[(curt-wt)] > 127.0f) )? 0.0 : background[(curb-wb)*3+1];
			fixed[curt*3+2] += ( (yb == 0) || (yt != 0 && mask[(curt-wt)] > 127.0f) )? 0.0 : background[(curb-wb)*3+2];

			//w_b
			fixed[curt*3+0] += ( (xb == 0) || (xt != 0 && mask[(curt-1)] > 127.0f) )? 0.0 : background[(curb-1)*3+0];
			fixed[curt*3+1] += ( (xb == 0) || (xt != 0 && mask[(curt-1)] > 127.0f) )? 0.0 : background[(curb-1)*3+1];
			fixed[curt*3+2] += ( (xb == 0) || (xt != 0 && mask[(curt-1)] > 127.0f) )? 0.0 : background[(curb-1)*3+2];

			//s_b
			fixed[curt*3+0] += ( (yb == hb-1) || (yt != ht-1 && mask[(curt+wt)] > 127.0f) )? 0.0 : background[(curb+wb)*3+0];
			fixed[curt*3+1] += ( (yb == hb-1) || (yt != ht-1 && mask[(curt+wt)] > 127.0f) )? 0.0 : background[(curb+wb)*3+1];
			fixed[curt*3+2] += ( (yb == hb-1) || (yt != ht-1 && mask[(curt+wt)] > 127.0f) )? 0.0 : background[(curb+wb)*3+2];

			//e_b
			fixed[curt*3+0] += ( (xb == wb-1) || (xt != wt-1 && mask[(curt+1)] > 127.0f) )? 0.0 : background[(curb+1)*3+0];
			fixed[curt*3+1] += ( (xb == wb-1) || (xt != wt-1 && mask[(curt+1)] > 127.0f) )? 0.0 : background[(curb+1)*3+1];
			fixed[curt*3+2] += ( (xb == wb-1) || (xt != wt-1 && mask[(curt+1)] > 127.0f) )? 0.0 : background[(curb+1)*3+2];
		}
	}
}

__global__ void PoissonImageCloningIteration(const float *fixed, const float *mask, const float *buf1, float *buf2, const int wt, const int ht) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		
		buf2[curt*3+0] = fixed[curt*3+0];
		buf2[curt*3+1] = fixed[curt*3+1];
		buf2[curt*3+2] = fixed[curt*3+2];

		//n_b
		buf2[curt*3+0] += (yt == 0 || mask[(curt-wt)] <= 127.0f)? 0.0 : buf1[(curt-wt)*3+0];
		buf2[curt*3+1] += (yt == 0 || mask[(curt-wt)] <= 127.0f)? 0.0 : buf1[(curt-wt)*3+1];
		buf2[curt*3+2] += (yt == 0 || mask[(curt-wt)] <= 127.0f)? 0.0 : buf1[(curt-wt)*3+2];

		//w_b
		buf2[curt*3+0] += (xt == 0 || mask[(curt-1)] <= 127.0f)? 0.0 : buf1[(curt-1)*3+0];
		buf2[curt*3+1] += (xt == 0 || mask[(curt-1)] <= 127.0f)? 0.0 : buf1[(curt-1)*3+1];
		buf2[curt*3+2] += (xt == 0 || mask[(curt-1)] <= 127.0f)? 0.0 : buf1[(curt-1)*3+2];

		//n_b
		buf2[curt*3+0] += (yt == ht-1 || mask[(curt+wt)] <= 127.0f)? 0.0 : buf1[(curt+wt)*3+0];
		buf2[curt*3+1] += (yt == ht-1 || mask[(curt+wt)] <= 127.0f)? 0.0 : buf1[(curt+wt)*3+1];
		buf2[curt*3+2] += (yt == ht-1 || mask[(curt+wt)] <= 127.0f)? 0.0 : buf1[(curt+wt)*3+2];

		//w_b
		buf2[curt*3+0] += (xt == wt-1 || mask[(curt+1)] <= 127.0f)? 0.0 : buf1[(curt+1)*3+0];
		buf2[curt*3+1] += (xt == wt-1 || mask[(curt+1)] <= 127.0f)? 0.0 : buf1[(curt+1)*3+1];
		buf2[curt*3+2] += (xt == wt-1 || mask[(curt+1)] <= 127.0f)? 0.0 : buf1[(curt+1)*3+2];

		buf2[curt*3+0] /= 4.0;
		buf2[curt*3+1] /= 4.0;
		buf2[curt*3+2] /= 4.0;
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	//printf("%d, %d, %d, %d, %d, %d\n", wb, hb, wt, ht, oy, ox);
	// set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	// initialize the iteration
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
	CaculateFixed<<<gdim, bdim>>>(background, target, mask, fixed, wb, hb, wt, ht, oy, ox);
	cudaMemcpy(buf1, target, sizeof(float)*3*ht*wt, cudaMemcpyDeviceToDevice);

	// iterate
	for (int i = 0; i < 10000; ++i) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, mask, buf1, buf2, wt, ht);
		PoissonImageCloningIteration<<<gdim, bdim>>>(fixed, mask, buf2, buf1, wt, ht);
	}


	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);

	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
