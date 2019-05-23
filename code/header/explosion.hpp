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
#define TITLE_STRING "BAKURETSU"

class Explosion {
  private:
    void eventFunctions();
    void initGLUT(int *argc, char **argv);
    void initBuffer();

  public:
    void start(int argc, char **argv);
    void display();
    Explosion();
};

#endif // ifndef EXPLOSION_HH
