#ifndef SHADER_H
#define SHADER_H

#include <fstream>
#include <string>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "GL/glew.h"

#define FRAGMENT_SHADER_DIR "./code/shaders/default.frag"
#define GEOMETRY_SHADER_DIR "./code/shaders/cube.geom"
#define VERTEX_SHADER_DIR "./code/shaders/default.vert"

GLuint LoadShader(const char *geometryShader, const char *vertexShader, const char *fragmentShader);
void loadUniforms(GLuint program);
#endif // ifndef SHADER_H
