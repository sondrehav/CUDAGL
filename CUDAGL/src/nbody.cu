#include "nbody.hpp"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define HANDLE_ERROR(x) \
	if(x != cudaSuccess) {\
		fprintf(stderr, "CUDA failed at line %d in file %s!\n", __LINE__, __FILE__);\
		fprintf(stderr, "Error: '%s\n'", _cudaGetErrorEnum(x));\
		system("pause"); \
		exit(-1);\
	}\

__global__ void add(Body *a, int num_items, double deltaTime)
{
	int thread_id = blockIdx.x;
	if (thread_id < N)
	{
		Body* thisBody = a + thread_id;
		float dx = thisBody->vx;
		float dy = thisBody->vy;
		for (int i = 0; i < num_items; i++)
		{
			if (i == thread_id) continue;
			Body otherBody = a[i];

			float dir_x = otherBody.x - thisBody->x;
			float dir_y = otherBody.y - thisBody->y;
			float lengthSq = dir_x * dir_x + dir_y * dir_y;
			float length = (float)sqrt(lengthSq);

			float nx = dir_x / length;
			float ny = dir_y / length;

			float force = G * thisBody->mass * otherBody.mass / lengthSq;

			float ddx = nx * force / thisBody->mass;
			float ddy = ny * force / thisBody->mass;

			dx += ddx * deltaTime;
			dy += ddy * deltaTime;

		}
		thisBody->vx = dx;
		thisBody->vy = dy;
	}
}

__global__ void updatePosition(Body *a, double deltaTime)
{
	int thread_id = blockIdx.x;
	if (thread_id < N)
	{
		Body* b = a + thread_id;
		b->x += b->vx * deltaTime;
		b->y += b->vy * deltaTime;
	}
}

void NBody::simulate(double deltaTime)
{
	Body* devPtr;
	size_t size;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &m_cudaVBOResource, NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, m_cudaVBOResource));

	int n_items = size / sizeof(Body);

	add<<<N, 1>>>(devPtr, n_items, deltaTime);
	updatePosition<<<N, 1>>>(devPtr, deltaTime);

	Body* cpuBody = new Body[n_items];
	HANDLE_ERROR(cudaMemcpy(cpuBody, devPtr, size, cudaMemcpyDeviceToHost));
	
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &m_cudaVBOResource, NULL));

	
}
