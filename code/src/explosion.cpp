#include "../header/explosion.hpp"

const unsigned int Explosion::numCubesX = 8;
const unsigned int Explosion::numCubesY = 8;
const unsigned int Explosion::numCubesZ = 8;
const unsigned int Explosion::numCubes  = numCubesX * numCubesY * numCubesZ;

const float Explosion::cubeSize = 12.0f;

GLuint Explosion::vbo           = 0;
float  Explosion::deltaTime     = 0;
bool   Explosion::setInitValues = false;
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
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numCubes * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

void Explosion::render() {
    float4 *ptr = 0;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, cuda_vbo_resource);
    if (!setInitValues) {
        vertexKernelLauncher(ptr, numCubesX, numCubesY, numCubesZ, cubeSize);
        setInitValues = true;
    }

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}

void Explosion::draw() {
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, 0.0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
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
    // GLSL init
    GLuint program = LoadShader(GEOMETRY_SHADER_DIR, VERTEX_SHADER_DIR, FRAGMENT_SHADER_DIR);
    glUseProgram(program);
    loadUniforms(program);
    // events init
    eventFunctions();
    // display func
    glutDisplayFunc(display);
    // Cuda init
    initBuffer();
    // main loop
    glutMainLoop();
}
