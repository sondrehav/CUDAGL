#include <cstdio>
#include "julia.hpp"
#include "julia_set.h"
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include <cassert>

int main(int argc, char** argv)
{
	Julia j;
	j.start(argc, argv);
}

Julia::~Julia(){}

void Julia::onInit(int width, int height, int argc, char** argv)
{

	GLuint vbo;
	glGenBuffers(1, &vbo);

	GLenum err = glGetError();
	if(err != GL_NO_ERROR)
	{
		printf("GL Error: %d\n", err);
	}

	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// initialize buffer object
	unsigned int size = width * height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	struct cudaGraphicsResource *cuda_vbo_resource;
	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));
}

void Julia::onResize(int width, int height)
{
	printf("Window resize! %d, %d\n", width, height);
}

void Julia::onLoop()
{
	
}

void Julia::onClose()
{
	printf("Close!");
}

void Julia::onKey(int key, int scancode, int action, int mods)
{
	printf("Key! %d, %d, %d, %d\n", key, scancode, action, mods);
	if(key == GLFW_KEY_ESCAPE)
	{
		this->close();
	}

	if(key == GLFW_KEY_SPACE)
	{
		test();
	}
}

void Julia::onMouseMove(double xpos, double ypos)
{
	printf("Mouse move! %f, %f\n", xpos, ypos);
}

void Julia::onMouseButton(int button, int action, int mods)
{
	printf("Mouse button! %d, %d, %d\n", button, action, mods);
}