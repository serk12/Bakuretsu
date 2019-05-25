#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 8) out;

out vec4 fColor;

//http://www.cs.umd.edu/gvil/papers/av_ts.pdf
void build_cube(vec4 position)
{
    float r = 0.2;
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

    gl_Position = p7;
    fColor = vec4(1.0, 0.0, 0.0, 0.0);
    EmitVertex();

    gl_Position = p8;
    fColor = vec4(0.0, 1.0, 0.0, 0.0);
    EmitVertex();

    gl_Position = p5;
    fColor = vec4(0.0, 0.0, 1.0, 0.0);
    EmitVertex();

    gl_Position = p6;
    fColor = vec4(1.0, 1.0, 0.0, 0.0);
    EmitVertex();

    gl_Position = p2;
    fColor = vec4(0.0, 1.0, 1.0, 0.0);
    EmitVertex();

    gl_Position = p8;
    fColor = vec4(1.0, 0.0, 1.0, 0.0);
    EmitVertex();

    gl_Position = p4;
    fColor = vec4(1.0, 1.0, 1.0, 0.0);
    EmitVertex();

    gl_Position = p7;
    fColor = vec4(0.0, 0.0, 0.5, 0.0);
    EmitVertex();

    gl_Position = p3;
    fColor = vec4(0.0, 0.5, 0.0, 0.0);
    EmitVertex();

    gl_Position = p5;
    fColor = vec4(0.5, 0.0, 0.0, 0.0);
    EmitVertex();

    gl_Position = p1;
    fColor = vec4(0.0, 0.5, 0.5, 0.0);
    EmitVertex();

    gl_Position = p2;
    fColor = vec4(0.5, 0.5, 0.0, 0.0);
    EmitVertex();

    gl_Position = p3;
    fColor = vec4(0.5, 0.0, 0.5, 0.0);
    EmitVertex();

    gl_Position = p4;
    fColor = vec4(0.5, 0.5, 0.5, 0.0);
    EmitVertex();
}

void main() {
    build_cube(gl_in[0].gl_Position);
}
