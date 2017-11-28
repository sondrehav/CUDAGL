#pragma once

#include <string>
#include <GLEW/glew.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#ifndef _DEBUG
#define HANDLE_CUDA_ERROR(x) x;
#endif

#ifndef _DEBUG
#define HANDLE_GL_ERROR(x) x;
#endif

#ifdef _DEBUG
#define HANDLE_CUDA_ERROR(x) \
	{\
	x; \
	{ cudaError_t __cudaError = cudaGetLastError(); \
	if(__cudaError != cudaSuccess) { \
		printf("CUDA failed at line %d in file %s!\n", __LINE__, __FILE__);\
		printf("Error: %s\n", _cudaGetErrorEnum(__cudaError));\
		printf("Call: '%s'\n", #x); \
		__debugbreak(); \
	}}\
	}
#endif

#ifdef _DEBUG
#define HANDLE_GL_ERROR(x) \
	x; \
	{\
	int __errorID__ = glGetError(); \
	if (__errorID__ != GL_NO_ERROR) { \
		printf("OpenGL failed at line %d in file %s!\nError: %s\n", __LINE__, __FILE__, glErrorString(__errorID__).c_str()); \
		printf("Call: '%s'\n", #x); \
		__debugbreak(); \
	}\
	}
#endif

inline std::string glErrorString(int errorCode) {
	switch (errorCode) {
	case GL_INVALID_ENUM:
		return "GL_INVALID_ENUM";
	case GL_INVALID_OPERATION:
		return "GL_INVALID_OPERATION";
	case GL_INVALID_FRAMEBUFFER_OPERATION:
		return "GL_INVALID_FRAMEBUFFER_OPERATION";
	case GL_OUT_OF_MEMORY:
		return "GL_OUT_OF_MEMORY";
	case GL_STACK_UNDERFLOW:
		return "GL_STACK_UNDERFLOW";
	case GL_STACK_OVERFLOW:
		return "GL_STACK_OVERFLOW";
	default:
		return "[Unknown error ID]";
	}
}