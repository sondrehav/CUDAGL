#include <cstdio>
#include "nbody.hpp"
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include "file.h"

int main(int argc, char** argv)
{
	NBody j;
	j.start(argc, argv);
}

NBody::~NBody(){}

void NBody::onInit(int width, int height, int argc, char** argv)
{

	printf("size: %zu\n", sizeof(Body));
	glGenVertexArrays(1, &m_vao);
	glBindVertexArray(m_vao);

	glGenBuffers(1, &m_vbo);

	GLenum err = glGetError();
	if(err != GL_NO_ERROR)
	{
		printf("GL Error: %d\n", err);
	}

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

	// initialize buffer object
	Body array[N];
	for(int i = 0; i < N; i++)
	{
		array[i].x = (2.0 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 1.0);
		array[i].y = (2.0 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 1.0);
		array[i].vx = 0.0;
		array[i].vy = 0.0;
		array[i].mass = (2.0 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + 0.5;
	}

	glBufferData(GL_ARRAY_BUFFER, sizeof(Body) * N, array, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Body), (void*)offsetof(Body, x));
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVBOResource, m_vbo, cudaGraphicsMapFlagsNone));

	m_shader = new Shader(readFile("shaders/shader.vs"), readFile("shaders/shader.fs"));
	float viewMatrix[16] = { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

	glUseProgram(m_shader->getProgramID());
	glUniformMatrix4fv(glGetUniformLocation(m_shader->getProgramID(), "view"), 1, false, viewMatrix);
	glUseProgram(0);

	glPointSize(256.0f);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_POINT_SPRITE);

	glViewport(0, 0, width, height);

	glBindVertexArray(m_vao);
	glUseProgram(m_shader->getProgramID());

}

void NBody::onResize(int width, int height)
{
	glViewport(0, 0, width, height);
}

void NBody::onLoop()
{
	
	if(m_simulate)
	{
		this->simulate(this->getFrameTime());
	}
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glDrawArrays(GL_POINTS, 0, N);

}

void NBody::onClose()
{
	printf("Close!");
	delete m_shader;
}

void NBody::onKey(int key, int scancode, int action, int mods)
{
	
	if(key == GLFW_KEY_ESCAPE)
	{
		this->close();
	}
	if(key == GLFW_KEY_SPACE && action == GLFW_PRESS)
	{
		this->m_simulate = !this->m_simulate;
	}
}



