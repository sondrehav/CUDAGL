#include <cstdio>
#include "nbody.hpp"
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include "file.h"
#include <random>
#include "error_check.h"

int main(int argc, char** argv)
{
	NBody j;
	j.start(argc, argv);
}

NBody::~NBody(){}

void NBody::onInit(int width, int height, int argc, char** argv)
{

	HANDLE_GL_ERROR(glGenVertexArrays(1, &m_vao));
	HANDLE_GL_ERROR(glBindVertexArray(m_vao));
	
	HANDLE_GL_ERROR(glGenBuffers(1, &m_vbo));

	HANDLE_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, m_vbo));

	HANDLE_GL_ERROR(glBufferData(GL_ARRAY_BUFFER, sizeof(Body) * N, NULL, GL_DYNAMIC_DRAW));

	HANDLE_GL_ERROR(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Body), (void*)offsetof(Body, x)));
	HANDLE_GL_ERROR(glEnableVertexAttribArray(0));
	HANDLE_GL_ERROR(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Body), (void*)offsetof(Body, vx)));
	HANDLE_GL_ERROR(glEnableVertexAttribArray(1));
	HANDLE_GL_ERROR(glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Body), (void*)offsetof(Body, mass)));
	HANDLE_GL_ERROR(glEnableVertexAttribArray(2));

	HANDLE_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
	HANDLE_GL_ERROR(glBindVertexArray(0));

	// register this buffer object with CUDA
	HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&m_cudaVBOResource, m_vbo, cudaGraphicsMapFlagsNone));

	m_shader = new Shader(readFile("shaders/smooth_shader.vs"), readFile("shaders/smooth_shader.fs"));
	m_quickDrawShader = new Shader(readFile("shaders/quick_shader.vs"), readFile("shaders/quick_shader.fs"));
	
	float viewMatrix[16] = { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, m_zoom };

	this->setZoom();
	this->useStandardShader();

	HANDLE_GL_ERROR(glViewport(0, 0, width, height));

	HANDLE_GL_ERROR(glBindVertexArray(m_vao));
	HANDLE_GL_ERROR(glEnable(GL_VERTEX_PROGRAM_POINT_SIZE));

	this->reset();

}

inline float random()
{
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void NBody::reset()
{
	HANDLE_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, m_vbo));

	// initialize buffer object
	Body array[N];
	
	std::random_device rd;
	std::mt19937 e2(rd());
	std::normal_distribution<> dist(0.0, 5.0);
	
	for (int i = 0; i < N; i++)
	{
		array[i].x = dist(e2);
		array[i].y = dist(e2);
		array[i].vx = random() * .2 - .1;
		array[i].vy = random() * .2 - .1;
		array[i].mass = random() * random() * random() * 10.0 + 1.0;
	}
	
	HANDLE_GL_ERROR(glBufferData(GL_ARRAY_BUFFER, sizeof(Body) * N, array, GL_DYNAMIC_DRAW));

}

void NBody::onResize(int width, int height)
{
	HANDLE_GL_ERROR(glViewport(0, 0, width, height));
}

void NBody::onLoop()
{
	
	if(m_simulate)
	{
		this->simulate(this->getFrameTime());
	}
	HANDLE_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
	HANDLE_GL_ERROR(glDrawArrays(GL_POINTS, 0, N));

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
	if (key == GLFW_KEY_A && action == GLFW_PRESS)
	{
		this->m_quickDraw = !this->m_quickDraw;
		if(this->m_quickDraw)
		{
			this->useQuickDrawShader();
		} else
		{
			this->useStandardShader();
		}
	}
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
	{
		this->m_simulate = !this->m_simulate;
	}
	if (key == GLFW_KEY_R && action == GLFW_PRESS)
	{
		this->reset();
	}
	if (key == GLFW_KEY_KP_ADD)
	{
		m_zoom *= 0.9;
		this->setZoom();
	}
	if (key == GLFW_KEY_KP_SUBTRACT)
	{
		m_zoom /= 0.9;
		this->setZoom();
	}
}

void NBody::setZoom()
{
	float viewMatrix[16] = { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, m_zoom };

	HANDLE_GL_ERROR(glUseProgram(m_shader->getProgramID()));
	HANDLE_GL_ERROR(glUniformMatrix4fv(glGetUniformLocation(m_shader->getProgramID(), "view"), 1, false, viewMatrix));
	HANDLE_GL_ERROR(glUseProgram(0));

	HANDLE_GL_ERROR(glUseProgram(m_quickDrawShader->getProgramID()));
	HANDLE_GL_ERROR(glUniformMatrix4fv(glGetUniformLocation(m_quickDrawShader->getProgramID(), "view"), 1, false, viewMatrix));
	HANDLE_GL_ERROR(glUseProgram(0));

	if (this->m_quickDraw) this->useQuickDrawShader();
	else this->useStandardShader();
}


void NBody::useQuickDrawShader()
{
	glDisable(GL_BLEND);
	glDisable(GL_POINT_SPRITE);
	glUseProgram(this->m_quickDrawShader->getProgramID());
}

void NBody::useStandardShader()
{
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_POINT_SPRITE);
	glUseProgram(this->m_shader->getProgramID());
}
