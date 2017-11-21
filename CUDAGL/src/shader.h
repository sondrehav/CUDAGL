#pragma once
#include <string>
#include <GLEW/glew.h>

class Shader
{
public:
	Shader(const std::string& vertexSource, const std::string& fragmentSource);

	inline GLuint getProgramID() const { return m_program; }

private:
	GLuint m_program;
	void checkShaderError(GLuint shader, GLenum flag, bool isProgram, const char* errorMessage);
	GLuint createShader(const std::string &source, GLenum shaderType);

};
