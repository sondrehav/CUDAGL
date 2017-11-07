#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 128

#define CHND(x) \
	if(x != cudaSuccess) {\
		fprintf(stderr, "CUDA failed at line %d in file %s!", __LINE__, __FILE__);\
		exit(-1);\
	}\

__global__ void add(float *a, float *b, float *out)
{
	int thread_id = blockIdx.x;
	if (thread_id < N)
	{
		out[thread_id] = a[thread_id] + b[thread_id];
	}
}

int test()
{
	srand(time(NULL));
	float a[N], b[N], out[N];
	float* dev_a, *dev_b, *dev_out;

	CHND(cudaMalloc((void**)&dev_a, N * sizeof(float)));
	CHND(cudaMalloc((void**)&dev_b, N * sizeof(float)));
	CHND(cudaMalloc((void**)&dev_out, N * sizeof(float)));

	for (int i = 0; i < N; i++)
	{
		a[i] = (float)rand() / (float)RAND_MAX;
		b[i] = (float)rand() / (float)RAND_MAX;
	}

	CHND(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	CHND(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

	add<<<N, 1>>>(dev_a, dev_b, dev_out);

	CHND(cudaMemcpy(out, dev_out, N * sizeof(float), cudaMemcpyDeviceToHost));

	printf("[ ");
	for (int i = 0; i < N; i++)
	{
		printf("%3.3f, ", out[i]);
	}
	printf(" ]\n");

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_out);

	return 0;

}
