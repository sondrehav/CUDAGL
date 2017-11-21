#include "shader.h"

Shader::Shader(const std::string& vertexSource, const std::string& fragmentSource)
{
	m_program = glCreateProgram();

	GLuint vertexShader = createShader(vertexSource, GL_VERTEX_SHADER);
	GLuint fragmentShader = createShader(fragmentSource, GL_FRAGMENT_SHADER);

	glAttachShader(m_program, vertexShader);
	glAttachShader(m_program, fragmentShader);

	glLinkProgram(m_program);
	checkShaderError(m_program, GL_LINK_STATUS, true, "ERROR: Program linking failed: ");

	glValidateProgram(m_program);
	checkShaderError(m_program, GL_VALIDATE_STATUS, true, "ERROR: Program is invalid: ");

}

void Shader::checkShaderError(GLuint shader, GLenum flag, bool isProgram, const char* errorMessage)
{
	GLint success = 0;
	GLchar errorlog[1024] = { 0 };
	if (isProgram) glGetProgramiv(shader, flag, &success);
	else glGetShaderiv(shader, flag, &success);
	if (success == GL_FALSE) {
		if (isProgram)
			glGetProgramInfoLog(shader, sizeof(errorlog), NULL, errorlog);
		else
			glGetShaderInfoLog(shader, sizeof(errorlog), NULL, errorlog);
		printf("%s\n", errorMessage);
		system("pause");
	}
}

GLuint Shader::createShader(const std::string &source, GLenum shaderType) {

	GLint shaderid = glCreateShader(shaderType);
	if (shaderid == 0) printf("ERROR: Shader creation failed!\n");
	const GLchar* shaderSource[1];
	GLint shaderSourceLength[1];
	shaderSource[0] = source.c_str();
	shaderSourceLength[0] = source.length();
	glShaderSource(shaderid, 1, shaderSource, shaderSourceLength);
	glCompileShader(shaderid);
	checkShaderError(shaderid, GL_COMPILE_STATUS, false, "ERROR: Shader failed to compile: ");
	return shaderid;

}