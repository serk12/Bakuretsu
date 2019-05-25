#include "../header/interactions.hpp"

Interactions::InteractiveAction Interactions::DoingInteractive = NONE;
int Interactions::xClick                                       = 0;
int Interactions::yClick                                       = 0;

void Interactions::keyboard(unsigned char key, int x, int y) {
    if (key == 27) exit(0);
    if (key == 'o') {
        perspectiva = !perspectiva;
        projecTransform();
    }
    if (key == 's') {
        scale += 0.05;
        modelTransform();
    }
    if (key == 'd') {
        scale -= 0.05;
        modelTransform();
    }
    if (key == 'r') {
        ++rotate;
        modelTransform();
    }
    if (key == 'z') {
        zoom /= 1.1f;
        projecTransform();
    }
    if (key == 'x') {
        zoom *= 1.1f;
        projecTransform();
    }
    if (key == 'l') DoingInteractive = ROTATEX;
    else if (key == 'k') DoingInteractive = ROTATEY;
    else if (key == 'i') DoingInteractive = ROTATEZ;
    glutPostRedisplay();
}

void Interactions::handleSpecialKeypress(int key, int x, int y) {}

void Interactions::mouseDrag(int x, int y) {}

void Interactions::mouseMove(int x, int y) {
    if (DoingInteractive == ROTATEX) {
        angleX += (x - xClick) * M_PI / 180.0;
        viewTransform();
    }
    else if (DoingInteractive == ROTATEY) {
        angleY += (x - xClick) * M_PI / 180.0;
        viewTransform();
    }
    else if (DoingInteractive == ROTATEZ) {
        angleZ += (x - xClick) * M_PI / 180.0;
        viewTransform();
    }
    DoingInteractive = NONE;

    xClick = x;
    yClick = y;
    glutPostRedisplay();
}


void Interactions::reshape(int w, int h) {
    ra = float(w / float(h));
    projecTransform();
    glViewport(0, 0, w, h);
}

void Interactions::animation() {
    Explosion::deltaTime += 0.01;
    Explosion::display();
}

void Interactions::exitfunc() {
    if (Explosion::vbo) {
        cudaGraphicsUnregisterResource(Explosion::cuda_vbo_resource);
        glDeleteBuffers(1, &Explosion::vbo);
    }
}

void Interactions::printInstructions() {
    printf("camera interactions\n");
    printf("esc: close graphics window\n");
}
