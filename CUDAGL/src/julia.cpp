#include "program.hpp"
#include <GLFW/glfw3.h>
#include <cstdio>

void onInit(GLFWwindow* window, int width, int height)
{
	printf("Init!\n");
}

void onResize(GLFWwindow* window, int width, int height)
{
	printf("Window resize! %d, %d\n", width, height);
}

void onLoop()
{
	
}

void onClose(GLFWwindow* window)
{
	printf("Close!");
}

void onKey(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	printf("Key! %d, %d, %d, %d\n", key, scancode, action, mods);
}

void onMouseMove(GLFWwindow* window, double xpos, double ypos)
{
	printf("Mouse move! %f, %f\n", xpos, ypos);
}

void onMouseButton(GLFWwindow* window, int button, int action, int mods)
{
	printf("Mouse button! %d, %d, %d\n", button, action, mods);
}