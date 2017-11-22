#pragma once
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include "program.hpp"
#include "shader.h"

#define N 2048
#define BLOCK_SIZE N
#define THREAD_SIZE 1024

#define G 0.001
#define DAMPING 0.99
#define RADIUS 0.1

struct Body
{
	float x, y;
	float vx, vy;
	float mass;
};

class NBody : public Program
{

public:

	~NBody() override;

	void onInit(int width, int height, int argc, char** argv) override;
	void onResize(int width, int height) override;
	void onLoop() override;
	void onClose() override;
	void onKey(int key, int scancode, int action, int mods) override;
	
	void simulate(double deltaTime);
	
	void useQuickDrawShader();
	void useStandardShader();
	void reset();
	void setZoom();

private:
	Shader* m_shader;
	Shader* m_quickDrawShader;
	GLuint m_vao;
	GLuint m_vbo;
	struct cudaGraphicsResource *m_cudaVBOResource;
	bool m_simulate = false;
	bool m_quickDraw = false;
	float m_zoom = 10.0;

};

