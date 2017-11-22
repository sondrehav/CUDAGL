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

__global__ void collision(Body *pnt, double deltaTime)
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	Body* a = thread_id + pnt;
	
	if (thread_id < N)
	{
		
		for(int i = thread_id + 1; i < N; i++)
		{
			
			Body* b = pnt + i;

			float a_x = a->x;
			float a_y = a->y;

			float b_x = b->x;
			float b_y = b->y;

			float dist_x = b_x - a_x;
			float dist_y = b_y - a_y;

			float dSq = dist_x * dist_x + dist_y * dist_y;

			if (dSq < 2 * RADIUS * RADIUS) {

				float a_dx = a->vx;
				float a_dy = a->vy;

				float b_dx = b->vx;
				float b_dy = b->vy;

				float a_m = a->mass;
				float b_m = b->mass;

				float length = sqrt(dSq);
				float n_x = dist_x / length;
				float n_y = dist_y / length;
				
				float a_d_dot_n = a_dx * n_x + a_dy * n_y;
				float b_d_dot_n = b_dx * n_x + b_dy * n_y;

				float totalMomentum = 2 * (a_d_dot_n - b_d_dot_n) / (a_m + b_m);

				float pn_x = n_x * totalMomentum;
				float pn_y = n_y * totalMomentum;

				float new_a_d_x = - pn_x * b_m;
				float new_a_d_y = - pn_y * b_m;

				float new_b_d_x = pn_x * a_m;
				float new_b_d_y = pn_y * a_m;

				a->vx += new_a_d_x;
				a->vy += new_a_d_y;

				b->vx += new_b_d_x;
				b->vy += new_b_d_y;

			}


		}
	}
}

__global__ void gravity(Body *pnt, double deltaTime)
{
	
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	Body* a = thread_id + pnt;
	
	float dx = 0.0;
	float dy = 0.0;

	if (thread_id < N)
	{
		for (int i = 0; i < N; i++)
		{
			if (i == thread_id) continue;
			Body* otherBody = pnt + i;

			float dir_x = otherBody->x - a->x;
			float dir_y = otherBody->y - a->y;
			float lengthSq = dir_x * dir_x + dir_y * dir_y;

			if (lengthSq < 2 * RADIUS * RADIUS) continue;

			float length = (float)sqrt(lengthSq);



			float nx = dir_x / length;
			float ny = dir_y / length;

			float force = G * a->mass * otherBody->mass / lengthSq;

			float ddx = nx * force / a->mass;
			float ddy = ny * force / a->mass;

			dx += ddx * deltaTime;
			dy += ddy * deltaTime;
		}
		a->vx += dx;
		a->vy += dy;
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
		b->vx *= DAMPING;
		b->vy *= DAMPING;
	}
}

void NBody::simulate(double deltaTime)
{


	Body* devPtr;
	size_t size;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &m_cudaVBOResource, NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, m_cudaVBOResource));

	collision<<<BLOCK_SIZE, THREAD_SIZE >>>(devPtr, deltaTime);
	cudaDeviceSynchronize();
	
	gravity<<<BLOCK_SIZE, THREAD_SIZE>>>(devPtr, deltaTime);
	cudaDeviceSynchronize();

	updatePosition<<<BLOCK_SIZE, THREAD_SIZE >>>(devPtr, deltaTime);
	cudaDeviceSynchronize();
	
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &m_cudaVBOResource, NULL));

	
}
