#version 330 core
out vec4 FragColor;  

in vec2 pass_position;
  
void main()
{
	float intensity = max(1.0 - length(2.0 * (gl_PointCoord.xy - vec2(0.5))), 0.0);
    FragColor = vec4(vec3(1.0), intensity * 0.1);
}	