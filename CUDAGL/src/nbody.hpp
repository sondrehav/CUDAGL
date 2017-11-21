#pragma once
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include "program.hpp"
#include "shader.h"

#define N 1000
#define G 0.000001

#pragma pack(1)
struct Body
{
	float x, y;
	float vx, vy;
	float mass;
};
#pragma pack(pop)

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

private:
	Shader* m_shader;
	GLuint m_vao;
	GLuint m_vbo;
	struct cudaGraphicsResource *m_cudaVBOResource;
	bool m_simulate = false;

};

