#include "../header/explosion.hpp"

const GLfloat Explosion::cubeRad    = 1.0f;
const GLfloat Explosion::bigCubeRad = (numCubesY / 8.0f) * 20.0f;

GLuint Explosion::vbo_pos       = 0;
GLuint Explosion::vbo_base_cube = 0;
GLuint Explosion::vbo_vel       = 0;
float  Explosion::deltaTime     = 0;
float  Explosion::vertices[4]   = { -bigCubeRad / 2.0f, -bigCubeRad / 2.0f, -bigCubeRad / 2.0f, 1.0f };

bool Explosion::setInitValues = false;
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
    glutInitWindowPosition(10, 10);
    glutInitWindowSize(W, H);
    glutCreateWindow(TITLE_STRING);
    glewInit();
    glClearColor(0, 0, 0, 0);
}

void Explosion::initBuffer() {
    glGenBuffers(1, &vbo_pos);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
    glBufferData(GL_ARRAY_BUFFER, numCubes * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_pos_resource, vbo_pos, cudaGraphicsMapFlagsWriteDiscard);

    glGenBuffers(1, &vbo_vel);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vel);
    glBufferData(GL_ARRAY_BUFFER, numCubes * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_vel_resource, vbo_vel, cudaGraphicsMapFlagsWriteDiscard);

    glGenBuffers(1, &vbo_base_cube);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_base_cube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}

void Explosion::render() {
    cudaGraphicsMapResources(1, &cuda_vbo_pos_resource, 0);
    cudaGraphicsMapResources(1, &cuda_vbo_vel_resource, 0);
    float4 *ptr_pos = 0, *ptr_vel = 0;
    size_t  num_bytes_pos, num_bytes_vel;
    cudaGraphicsResourceGetMappedPointer((void **)&ptr_pos, &num_bytes_pos, cuda_vbo_pos_resource);
    cudaGraphicsResourceGetMappedPointer((void **)&ptr_vel, &num_bytes_vel, cuda_vbo_vel_resource);

    if (!setInitValues) {
        initCubesDataKernal(ptr_pos, ptr_vel);
        setInitValues = true;
    }
    else {
        cubesUpdate(ptr_pos, ptr_vel, float(bigCubeRad), deltaTime);
    }

    cudaGraphicsUnmapResources(1, &cuda_vbo_pos_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_vbo_vel_resource, 0);
}

void Explosion::draw() {
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    setRadius(cubeRad);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, numCubes);
    glDisableClientState(GL_VERTEX_ARRAY);

    setRadius(bigCubeRad);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_base_cube);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, 1);
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
    loadUniforms(program, cubeRad, bigCubeRad);
    // events init
    eventFunctions();
    // display func
    glutDisplayFunc(display);
    // Cuda init
    initBuffer();
    // main loop
    glutMainLoop();
}
