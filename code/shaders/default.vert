#version 330 core

layout(location = 0) in vec4 vert;
layout (location = 1) in vec4 aColor;

uniform mat4 TG;
uniform mat4 proj;
uniform mat4 view;

out vec4 color;

void main()
{
    color = aColor;
    gl_Position = proj * view * TG * vert;
}
