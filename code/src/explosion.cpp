#include "../header/explosion.hpp"

Explosion *x;
extern "C" void displayTrampoline()
{
    return x->display();
}

Explosion::Explosion() {
    x = this;
}

void Explosion::eventFunctions() {
    glutKeyboardFunc(Interactions::keyboard);
    glutSpecialFunc(Interactions::handleSpecialKeypress);
    glutPassiveMotionFunc(Interactions::mouseMove);
    glutMotionFunc(Interactions::mouseDrag);
    glutReshapeFunc(Interactions::reshape);
    glutIdleFunc(Interactions::animation);
    atexit(Interactions::exitfunc);
}

void Explosion::initGLUT(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(600, 600);
    glutCreateWindow(TITLE_STRING);
    glClearColor(0, 0, 0, 0);
}

void Explosion::initBuffer() {}

void Explosion::display() {}

void Explosion::start(int argc, char **argv) {
    // show tutorial
    Interactions::printInstructions();
    // OpenGL init
    initGLUT(&argc, argv);
    // events init
    eventFunctions();
    // display func
    glutDisplayFunc(displayTrampoline);
    // Cuda init
    initBuffer();
    // main loop
    glutMainLoop();
}
