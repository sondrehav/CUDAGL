#pragma once

#include <GLFW/glfw3.h>

#define WIDTH 640
#define HEIGHT 480

void onInit(GLFWwindow* window, int width, int height);

void onResize(GLFWwindow* window, int width, int height);

void onLoop();

void onClose(GLFWwindow* window);

void onKey(GLFWwindow* window, int key, int scancode, int action, int mods);

void onMouseMove(GLFWwindow* window, double xpos, double ypos);

void onMouseButton(GLFWwindow* window, int button, int action, int mods);
