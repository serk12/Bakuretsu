#include "../header/interactions.hpp"

Interactions::InteractiveAction Interactions::DoingInteractive = NONE;
int Interactions::xClick                                       = 0;
int Interactions::yClick                                       = 0;
int Interactions::oldTime                                      = 0;

void Interactions::keyboard(unsigned char key, int x, int y) {
    if (key == 27) exit(0);
    if (key == 'o') {
        perspectiva = !perspectiva;
        projecTransform();
        updateViewProjectMatrix();
    }
    if (key == 's') {
        scale += 0.05;
        modelTransform();
        updateViewProjectMatrix();
    }
    if (key == 'd') {
        scale -= 0.05;
        modelTransform();
        updateViewProjectMatrix();
    }
    if (key == 'r') {
        ++rotate;
        modelTransform();
        updateViewProjectMatrix();
    }
    if (key == 'z') {
        zoom /= 1.1f;
        projecTransform();
        updateViewProjectMatrix();
    }
    if (key == 'x') {
        zoom *= 1.1f;
        projecTransform();
        updateViewProjectMatrix();
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
        updateViewProjectMatrix();
    }
    else if (DoingInteractive == ROTATEY) {
        angleY += (x - xClick) * M_PI / 180.0;
        viewTransform();
        updateViewProjectMatrix();
    }
    else if (DoingInteractive == ROTATEZ) {
        angleZ += (x - xClick) * M_PI / 180.0;
        viewTransform();
        updateViewProjectMatrix();
    }
    DoingInteractive = NONE;

    xClick = x;
    yClick = y;
    glutPostRedisplay();
}


void Interactions::reshape(int w, int h) {
    ra = float(w / float(h));
    projecTransform();
    updateViewProjectMatrix();
    glViewport(0, 0, w, h);
}

void Interactions::animation() {
    int newTime = glutGet(GLUT_ELAPSED_TIME);
    Explosion::deltaTime = (newTime - oldTime) / 1000.0f;
    oldTime              = newTime;
    Explosion::display();
}

void Interactions::exitfunc() {
    if (Explosion::vbo_vel) {
        cudaGraphicsUnregisterResource(Explosion::cuda_vbo_vel_resource);
        glDeleteBuffers(1, &Explosion::vbo_vel);
    }
    if (Explosion::vbo_pos) {
        cudaGraphicsUnregisterResource(Explosion::cuda_vbo_pos_resource);
        glDeleteBuffers(1, &Explosion::vbo_pos);
    }
    if (Explosion::vbo_base_cube) {
        glDeleteBuffers(1, &Explosion::vbo_base_cube);
    }
}

void Interactions::printInstructions() {
    printf("camera interactions\n");
    printf("esc: close graphics window\n");
    printf("i+mouse(x): rotate camara z axis\n");
    printf("k+mouse(x): rotate camara y axis\n");
    printf("l+mouse(x): rotate camara x axis\n");

    printf("o: changes camera perspective\n");

    printf("d: scale\n");
    printf("s: descale\n");

    printf("z: zoom in\n");
    printf("x: zoom out\n");

    printf("r: rotate x axis\n");
}
