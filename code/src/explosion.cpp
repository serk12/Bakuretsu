#include "../header/explosion.hpp"

Explosion *x;
extern "C" void displayTrampoline() {
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

void Explosion::initBuffer() {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numCubes * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

void Explosion::render() {
    float4 *ptr = 0;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, cuda_vbo_resource);

    vertexKernelLauncher(ptr, numCubes, deltaTime);

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}

void Explosion::draw() {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, numCubes);
    glDisableClientState(GL_VERTEX_ARRAY);
}

void Explosion::display() {
    render();
    draw();
    glutSwapBuffers();
}

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
