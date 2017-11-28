#version 330 core
layout (location = 0) in vec2 aPos;   // the position variable has attribute position 0
layout (location = 2) in float mass;   // the position variable has attribute position 0

uniform mat4 view = mat(1.0);

void main()
{
    gl_Position = view * vec4(aPos, 0.0, 1.0);
    gl_PointSize = 1;
}