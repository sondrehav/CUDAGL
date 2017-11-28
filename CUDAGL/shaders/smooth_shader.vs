#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aVel;
layout (location = 2) in float mass;
  
out vec2 pass_position; // output a color to the fragment shader
out float pass_mass;

uniform mat4 view = mat(1.0);

void main()
{
	gl_Position = view * vec4(aPos, 0.0, 1.0);
	gl_PointSize = 5.0;
    pass_position = aPos;
    pass_mass = mass;
}