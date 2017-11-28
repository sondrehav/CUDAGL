#version 330 core
out vec4 FragColor;

in vec2 pass_position;
in float pass_mass;

void main()
{

	float theta = mod(pass_mass, 1.0) * 2.0 * 3.14159;
	vec2 dir = vec2(cos(theta), sin(theta));

	vec3 color = vec3(
		max(dot(dir, vec2(1.0, 0.0)), 0.0),
		max(dot(dir, vec2(cos(3.14159 * 2.0 / 3.0), sin(3.14159 * 2.0 / 3.0))), 0.0),
		max(dot(dir, vec2(cos(-3.14159 * 2.0 / 3.0), sin(-3.14159 * 2.0 / 3.0))), 0.0)
	);
	color += 0.5;

	float intensity = max(1.0 - length(2.0 * (gl_PointCoord.xy - vec2(0.5))), 0.0);
    FragColor = vec4(color, intensity);
    //FragColor = vec4(1.0);
}	