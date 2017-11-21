#version 330 core
layout (location = 0) in vec2 aPos;   // the position variable has attribute position 0
  
out vec2 pass_position; // output a color to the fragment shader

uniform mat4 view = mat(1.0);

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    pass_position = aPos;
}