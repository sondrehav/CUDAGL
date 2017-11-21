#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include <GLFW/glfw3.h>
#include <string>
#include <stdlib.h>

#define WIDTH 640
#define HEIGHT 480

// Checks for whether an OpenGL error occurred. If one did,
// it prints out the error type and ID
inline void printGLError(int line, std::string file) {
	int errorID = glGetError();

	if (errorID != GL_NO_ERROR) {
		std::string errorString;

		switch (errorID) {
		case GL_INVALID_ENUM:
			errorString = "GL_INVALID_ENUM";
			break;
		case GL_INVALID_OPERATION:
			errorString = "GL_INVALID_OPERATION";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			errorString = "GL_INVALID_FRAMEBUFFER_OPERATION";
			break;
		case GL_OUT_OF_MEMORY:
			errorString = "GL_OUT_OF_MEMORY";
			break;
		case GL_STACK_UNDERFLOW:
			errorString = "GL_STACK_UNDERFLOW";
			break;
		case GL_STACK_OVERFLOW:
			errorString = "GL_STACK_OVERFLOW";
			break;
		default:
			errorString = "[Unknown error ID]";
			break;
		}

		printf("%s:%i: An OpenGL error occurred (%i): %s.\n", file.c_str(), line,
			errorID, errorString.c_str());
		system("pause");
	}
}

class Program
{

public:

	virtual ~Program() {}

	virtual void onInit(int width, int height, int argc = 0, char** argv = NULL) {}
	virtual void onResize(int width, int height) {}
	virtual void onLoop() {}
	virtual void onClose() {}
	virtual void onKey(int key, int scancode, int action, int mods) {}
	virtual void onMouseMove(double xpos, double ypos) {}
	virtual void onMouseButton(int button, int action, int mods) {}

	void start(int argc, char** argv);
	void close();

	static double getFrameTime() { return m_frameTime; }

private:

	static double m_frameTime;
	static double m_titleSet;

	static GLFWwindow* s_window;
	static Program* s_active;

	static void s_onResize(GLFWwindow* window, int width, int height);
	static void s_onKey(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void s_onMouseMove(GLFWwindow* window, double xpos, double ypos);
	static void s_onMouseButton(GLFWwindow* window, int button, int action, int mods);

	static bool initializeGLFW();

};

