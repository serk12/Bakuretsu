#include "../header/explosion.hpp"

const unsigned int Explosion::numCubes = 256 * 256;
GLuint Explosion::vbo                  = 0;
float  Explosion::deltaTime            = 0;
struct cudaGraphicsResource *Explosion::cuda_vbo_resource;

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
    glewInit();
    glClearColor(0, 0, 0, 0);
}

void Explosion::initBuffer() {
    glGenBuffers(1, &Explosion::vbo);
    glBindBuffer(GL_ARRAY_BUFFER, Explosion::vbo);
    glBufferData(GL_ARRAY_BUFFER, numCubes * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&Explosion::cuda_vbo_resource, Explosion::vbo, cudaGraphicsMapFlagsWriteDiscard);
}

void Explosion::render() {
    float4 *ptr = 0;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, cuda_vbo_resource);

    vertexKernelLauncher(ptr, numCubes, Explosion::deltaTime);

    cudaGraphicsUnmapResources(1, &Explosion::cuda_vbo_resource, 0);
}

void Explosion::draw() {
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, 0.0);

    glBindBuffer(GL_ARRAY_BUFFER, Explosion::vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, Explosion::numCubes);
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
    glutDisplayFunc(display);
    // Cuda init
    initBuffer();
    // main loop
    glutMainLoop();
}
