#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <thrust/replace.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


struct is_alphabet{
	__host__ __device__ 
	bool operator()(int x) {
		return x >= (int)'a' && x <= (int)'z';
	}
};


void CountPosition1(const char *text, int *pos, int text_size)
{
		thrust::replace_copy(thrust::device, text, text + text_size, pos, (int)'\n', 0);
		thrust::replace_if(thrust::device, pos, pos + text_size, is_alphabet(), 1);
		thrust::inclusive_scan_by_key(thrust::device, pos, pos + text_size, pos, pos);
}

void CountPosition2(const char *text, int *pos, int text_size)
{
}
