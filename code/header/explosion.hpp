#ifndef EXPLOSION_HH
#define EXPLOSION_HH

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif // ifdef _WIN32

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>

#include "./interactions.hpp"
#include "./cudaManager.h"
#include "./shader.hpp"

#define TITLE_STRING "BAKURETSU"

class Explosion {
  private:
    static void eventFunctions();
    static void initGLUT(int *argc, char **argv);
    static void initBuffer();
    static void render();
    static void draw();

    static const unsigned int numCubes, numCubesX, numCubesY, numCubesZ;
    static const float   cubeSize;
    static const GLfloat cubeRad;
    static bool setInitValues;
  public:
    // temporal fix
    static struct cudaGraphicsResource *cuda_vbo_pos_resource, *cuda_vbo_vel_resource;
    static GLuint vbo_pos, vbo_vel;
    static float  deltaTime;

    static void start(int argc, char **argv);
    static void display();
};

#endif // ifndef EXPLOSION_HH
