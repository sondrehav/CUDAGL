#include "shader.h"

#include "error_check.h"

Shader::Shader(const std::string& vertexSource, const std::string& fragmentSource)
{
	HANDLE_GL_ERROR(m_program = glCreateProgram());

	GLuint vertexShader = createShader(vertexSource, GL_VERTEX_SHADER);
	GLuint fragmentShader = createShader(fragmentSource, GL_FRAGMENT_SHADER);

	HANDLE_GL_ERROR(glAttachShader(m_program, vertexShader));
	HANDLE_GL_ERROR(glAttachShader(m_program, fragmentShader));

	HANDLE_GL_ERROR(glLinkProgram(m_program));
	checkShaderError(m_program, GL_LINK_STATUS, true, "ERROR: Program linking failed: ");

	HANDLE_GL_ERROR(glValidateProgram(m_program));
	checkShaderError(m_program, GL_VALIDATE_STATUS, true, "ERROR: Program is invalid: ");

}

void Shader::checkShaderError(GLuint shader, GLenum flag, bool isProgram, const char* errorMessage)
{
	GLint success = 0;
	GLchar errorlog[1024] = { 0 };
	if (isProgram) {
		HANDLE_GL_ERROR(glGetProgramiv(shader, flag, &success));
	}
	else {
		HANDLE_GL_ERROR(glGetShaderiv(shader, flag, &success));
	}
	if (success == GL_FALSE) {
		if (isProgram)
		{
			HANDLE_GL_ERROR(glGetProgramInfoLog(shader, sizeof(errorlog), NULL, errorlog));
		}
		else
		{
			HANDLE_GL_ERROR(glGetShaderInfoLog(shader, sizeof(errorlog), NULL, errorlog));
		}
		printf("Error: Shader failed to compile!\n");
		printf("%s\n", errorMessage);
		printf("%s\n", errorlog);
		__debugbreak();
	}
}

GLuint Shader::createShader(const std::string &source, GLenum shaderType) {

	HANDLE_GL_ERROR(GLint shaderid = glCreateShader(shaderType));
	if (shaderid == 0) {
		printf("ERROR: Shader creation failed!\n");
		__debugbreak();
	}
	const GLchar* shaderSource[1];
	GLint shaderSourceLength[1];
	shaderSource[0] = source.c_str();
	shaderSourceLength[0] = source.length();
	HANDLE_GL_ERROR(glShaderSource(shaderid, 1, shaderSource, shaderSourceLength));
	HANDLE_GL_ERROR(glCompileShader(shaderid));
	checkShaderError(shaderid, GL_COMPILE_STATUS, false, "ERROR: Shader failed to compile: ");
	return shaderid;

}