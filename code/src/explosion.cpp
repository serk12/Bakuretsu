#include "../header/explosion.hpp"

const unsigned int Explosion::numCubesX = 8;
const unsigned int Explosion::numCubesY = 8;
const unsigned int Explosion::numCubesZ = 8;
const unsigned int Explosion::numCubes  = numCubesX * numCubesY * numCubesZ;

const float   Explosion::cubeSize = numCubesY + 0.3f;
const GLfloat Explosion::cubeRad  = 1.0f;

GLuint Explosion::vbo_pos       = 0;
GLuint Explosion::vbo_vel       = 0;
float  Explosion::deltaTime     = 0;
bool   Explosion::setInitValues = false;
struct cudaGraphicsResource *Explosion::cuda_vbo_pos_resource, *Explosion::cuda_vbo_vel_resource;

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
    glGenBuffers(1, &vbo_pos);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
    glBufferData(GL_ARRAY_BUFFER, numCubes * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_pos_resource, vbo_pos, cudaGraphicsMapFlagsWriteDiscard);

    glGenBuffers(1, &vbo_vel);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vel);
    glBufferData(GL_ARRAY_BUFFER, numCubes * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_vel_resource, vbo_vel, cudaGraphicsMapFlagsWriteDiscard);
}

void Explosion::render() {
    cudaGraphicsMapResources(1, &cuda_vbo_pos_resource, 0);
    cudaGraphicsMapResources(1, &cuda_vbo_vel_resource, 0);
    float4 *ptr_pos = 0, *ptr_vel = 0;
    size_t  num_bytes_pos, num_bytes_vel;
    cudaGraphicsResourceGetMappedPointer((void **)&ptr_pos, &num_bytes_pos, cuda_vbo_pos_resource);
    cudaGraphicsResourceGetMappedPointer((void **)&ptr_vel, &num_bytes_vel, cuda_vbo_vel_resource);

    if (!setInitValues) {
        initCubesDataKernal(ptr_pos, ptr_vel, numCubesX, numCubesY, numCubesZ, cubeSize);
        setInitValues = true;
    }
    else {
        cubesUpdate(ptr_pos, ptr_vel, numCubes);
    }

    cudaGraphicsUnmapResources(1, &cuda_vbo_pos_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_vbo_vel_resource, 0);
}

void Explosion::draw() {
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
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
    loadUniforms(program, cubeRad);
    // events init
    eventFunctions();
    // display func
    glutDisplayFunc(display);
    // Cuda init
    initBuffer();
    // main loop
    glutMainLoop();
}
