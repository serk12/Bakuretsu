#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>

#include "./shader.hpp"
#include "./explosion.hpp"

class Interactions {
  private:
    typedef enum { NONE, ROTATEY, ROTATEX, ROTATEZ } InteractiveAction;
    static InteractiveAction DoingInteractive;
    static int xClick, yClick, oldTime;

  public:
    static void keyboard(unsigned char key, int x, int y);
    static void handleSpecialKeypress(int key, int x, int y);
    static void mouseMove(int x, int y);
    static void mouseDrag(int x, int y);
    static void reshape(int x, int y);
    static void animation();
    static void exitfunc();
    static void printInstructions();
};

#endif /* ifndef INTERACTIONS_H */
