#version 330 core
layout (location = 0) in vec2 aPos;   // the position variable has attribute position 0
layout (location = 1) in vec2 aVel;   // the position variable has attribute position 0
  
out vec2 pass_position; // output a color to the fragment shader
out vec4 pass_color;

uniform mat4 view = mat(1.0);

void main()
{

	vec2 dir = normalize(aVel);
	float red   = max(dot(vec2(1.0, 0.0), dir), 0.0);
	float green = max(dot(vec2(cos(2.094395), sin(2.094395)), dir), 0.0);
	float blue  = max(dot(vec2(cos(2.094395), sin(-2.094395)), dir), 0.0);

    gl_Position = view * vec4(aPos, 0.0, 1.0);
    pass_position = aPos;
    pass_color = vec4(red, green, blue, length(aVel));
}