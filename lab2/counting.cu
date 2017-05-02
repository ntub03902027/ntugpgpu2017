#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <cmath>
#include <vector>
#include <thrust/replace.h>
#include "SyncedMemory.h"
#define nThreads 1024
#define BinaryIndexTree

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


struct is_alphabet{
	__host__ __device__ 
	bool operator()(int x) {
		return x >= (int)'a' && x <= (int)'z';
	}
};

__global__ void fillNaive(const char *text, int *pos, const int text_size) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	int i = x;
	if (i < text_size)
		while (i >= 0 && text[i] != '\n')
			i--;
	pos[x] = x - i;
}

__global__ void copyFillTable(const char *text, int *binary_index_tree_table, const int text_size, const int treeLevel) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < text_size && text[x] != '\n') {
		binary_index_tree_table[x + (int)exp2((double)treeLevel)] = 1;
		binary_index_tree_table[x] = 0;
	}
	else if (x < (int)exp2((double)treeLevel)) {
		binary_index_tree_table[x + (int)exp2((double)treeLevel)] = 0;
		binary_index_tree_table[x] = 0;
	}
}
__global__ void fillTable(int *binary_index_tree_table, const int sub_table_size, const int start_index) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < sub_table_size && binary_index_tree_table[2*(start_index+x)] != 0 && binary_index_tree_table[2*(start_index+x)+1] != 0)
		binary_index_tree_table[x + start_index] = 2 * binary_index_tree_table[2*(start_index+x)];
}

__global__ void fillPos(int *pos, const int *binary_index_tree_table, const int text_size, const int treeLevel) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < text_size) {
		if (binary_index_tree_table[(int)exp2f((float)treeLevel) + x] == 0)
			pos[x] = 0;
		else {
			// maintain 4 values and 2 conditions
			bool done = false;
			bool traverse_down = false;
			int currentDivisor = 1;
			int currentAlign = x;
			int currentLength = 1;
			int currentIndex = ((int)exp2f((float)treeLevel) + x);
			currentDivisor *= 2;
			while (!done) {
				if (!traverse_down) {
					int newAlign = currentAlign;
					if (currentAlign % currentDivisor != 0)
						newAlign -= (currentDivisor / 2);
					if (currentAlign != newAlign) { // Not Already
						// index -> index - 1, check whether new index isn't 0	
						currentIndex--;
						if (binary_index_tree_table[currentIndex] != 0) { // Ok : Index / 2
							currentLength += binary_index_tree_table[currentIndex];
							currentAlign = newAlign;
							currentDivisor *= 2;
							currentIndex /= 2;
							if (currentIndex == 1)
								done = true;
						}
						else { // Fail, change to traverse_down, align no need to maintain anymore
							currentDivisor /= 4;
							currentIndex = currentIndex * 2 + 1;
							traverse_down = true;
							if (currentDivisor == 0)
								done = true;
						}
					}
					else { // Already
						currentDivisor *= 2;
						currentIndex /= 2;
						if (currentIndex == 1)
							done = true;
					}

				}
				else {
					if (binary_index_tree_table[currentIndex] != 0) {
						currentLength += binary_index_tree_table[currentIndex];
						currentDivisor /= 2;
						currentIndex = (currentIndex - 1) * 2 + 1;
					}
					else {
						currentDivisor /= 2;
						currentIndex = (currentIndex) * 2 + 1;
					}
					if (currentDivisor == 0)
						done = true;
				}
			}	
			pos[x] = currentLength;
		}

	}
}

void CountPosition1(const char *text, int *pos, int text_size)
{
	thrust::replace_copy(thrust::device, text, text + text_size, pos, (int)'\n', 0);
	thrust::replace_if(thrust::device, pos, pos + text_size, is_alphabet(), 1);
	thrust::inclusive_scan_by_key(thrust::device, pos, pos + text_size, pos, pos);
}

void CountPosition2(const char *text, int *pos, int text_size)
{

#ifndef BinaryIndexTree
	fillNaive<<<(text_size-1)/nThreads + 1, nThreads>>>(text, pos, text_size);
#else
	int treeLevel = (int)ceil(log2((float)text_size));
	int half_table_size = (int)exp2((double)treeLevel);

	std::vector<int> binary_index_tree_table_vec;
	binary_index_tree_table_vec.assign(2 * half_table_size, 0);

	int *binary_index_tree_table;
	cudaMalloc((void**)&binary_index_tree_table, 2 * half_table_size * sizeof(int));
	SyncedMemory<int> table_sync(binary_index_tree_table_vec.data(), binary_index_tree_table, binary_index_tree_table_vec.size());
	
	//cudaMemset(binary_index_tree_table, 0, sizeof(int)* 2 * (int)exp2((double)treeLevel));
	copyFillTable<<<(half_table_size-1)/nThreads + 1, nThreads>>>(text, table_sync.get_gpu_wo(), text_size, treeLevel);

	int multiplier = 2;
	for (int i = treeLevel - 1; i >= 0; --i) {
		fillTable<<<(half_table_size /multiplier -1) /nThreads + 1, nThreads>>>(table_sync.get_gpu_rw(), half_table_size / multiplier, (int)exp2((double)i));
		multiplier *= 2;
	}

	fillPos<<<(text_size-1)/nThreads + 1, nThreads>>>(pos, table_sync.get_gpu_ro(), text_size, treeLevel);


	cudaFree(binary_index_tree_table);
	
#endif
}
