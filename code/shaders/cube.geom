#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 14) out;

uniform mat4 TG;
uniform mat4 proj;
uniform mat4 view;
uniform float r;

out vec4 fColor;

//http://www.cs.umd.edu/gvil/papers/av_ts.pdf
void build_cube(vec4 position)
{
    vec4 dx = vec4(r,0.0,0.0,0.0);
    vec4 dy = vec4(0.0,r,0.0,0.0);
    vec4 dz = vec4(0.0,0.0,r,0.0);

    vec4 p1 = position;
    vec4 p2 = position + dx;
    vec4 p3 = position + dy;
    vec4 p4 = p2 + dy;
    vec4 p5 = p1 + dz;
    vec4 p6 = p2 + dz;
    vec4 p7 = p3 + dz;
    vec4 p8 = p4 + dz;

    gl_Position = proj * view * TG * p7;
    fColor = vec4(1.0, 0.0, 0.0, 0.0);
    EmitVertex();

    gl_Position =proj * view * TG * p8;
    EmitVertex();

    gl_Position = proj * view * TG *p5;
    EmitVertex();

    gl_Position = proj * view * TG *p6;
    fColor = vec4(0.0, 1.0, 0.0, 0.0);
    EmitVertex();

    gl_Position = proj * view * TG *p2;
    EmitVertex();

    gl_Position = proj * view * TG *p8;
    EmitVertex();

    gl_Position = proj * view * TG * p4;
    fColor = vec4(0.0, 0.0, 1.0, 0.0);
    EmitVertex();

    gl_Position = proj * view * TG * p7;
    EmitVertex();

    gl_Position = proj * view * TG * p3;
    EmitVertex();

    gl_Position = proj * view * TG * p5;
    fColor = vec4(0.0, 1.0, 1.0, 0.0);
    EmitVertex();

    gl_Position = proj * view * TG * p1;
    EmitVertex();

    gl_Position = proj * view * TG * p2;
    EmitVertex();

    gl_Position = proj * view * TG * p3;
    fColor = vec4(1.0, 0.0, 1.0, 0.0);
    EmitVertex();

    gl_Position = proj * view * TG * p4;
    EmitVertex();
}

void main() {
    build_cube(gl_in[0].gl_Position);
}
