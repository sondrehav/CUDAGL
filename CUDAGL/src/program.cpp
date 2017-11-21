
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>

#include "program.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <cassert>

GLFWwindow* Program::s_window = NULL;
Program* Program::s_active = NULL;

void Program::start(int argc, char** argv)
{

	assert(s_active == NULL);
	assert(initializeGLFW());
	s_active = this;

	this->onInit(WIDTH, HEIGHT, argc, argv);

	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(s_window))
	{
		/* Render here */
		this->onLoop();
		/* Swap front and back buffers */
		glfwSwapBuffers(s_window);

		/* Poll for and process events */
		glfwPollEvents();
	}

	this->onClose();

	glfwTerminate();
}

bool Program::initializeGLFW()
{
	GLFWwindow* window;

	/* Initialize the library */
	if (glfwInit() != GLFW_TRUE)
		return false;



	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(WIDTH, HEIGHT, "Hello World", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return false;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	/* Initialize GLEW */
	if(glewInit() != GLEW_OK)
	{
		printf("GLEW init error\n");
		glfwTerminate();
		return false;
	}

	glfwSetWindowSizeCallback(window, s_onResize);
	glfwSetKeyCallback(window, s_onKey);
	glfwSetCursorPosCallback(window, s_onMouseMove);
	glfwSetMouseButtonCallback(window, s_onMouseButton);

	// CUDA setup


	//cudaGLSetGLDevice
	int devID = findCudaDevice();
	assert(cudaGLSetGLDevice(devID) == cudaSuccess);

	s_window = window;
	
	return true;

}

void Program::close()
{
	glfwSetWindowShouldClose(s_window, GLFW_TRUE);
}

void Program::s_onResize(GLFWwindow* window, int width, int height) { s_active->onResize(width, height); }
void Program::s_onKey(GLFWwindow* window, int key, int scancode, int action, int mods) { s_active->onKey(key, scancode, action, mods); }
void Program::s_onMouseMove(GLFWwindow* window, double xpos, double ypos) { s_active->onMouseMove(xpos, ypos); }
void Program::s_onMouseButton(GLFWwindow* window, int button, int action, int mods) { s_active->onMouseButton(button, action, mods); }
