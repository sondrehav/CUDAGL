#pragma once
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include "program.hpp"
#include "shader.h"

#define N 2048
#define THREAD_SIZE 1024
#define BLOCK_SIZE N / THREAD_SIZE

#define G 0.01
#define DAMPING 0.95
#define RADIUS 0.1
#define MASS_DIST(x) (x * x * x * x * x * 50.0 + 1.0)

struct Body
{
	float x, y;
	float vx, vy;
	float mass;
};

class NBody : public program::Program
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

