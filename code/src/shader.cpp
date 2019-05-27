#include "../header/shader.hpp"

std::string readFile(const char *filePath) {
    std::string   content;
    std::ifstream fileStream(filePath, std::ios::in);

    if (!fileStream.is_open()) {
        std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
        return "";
    }

    std::string line = "";
    while (!fileStream.eof()) {
        std::getline(fileStream, line);
        content.append(line + "\n");
    }

    fileStream.close();
    return content;
}
GLuint procesShaderFile(const char *shaderPath, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);

    // Read shaders
    std::string shaderStr = readFile(shaderPath);
    const char *shaderSrc = shaderStr.c_str();

    GLint result = GL_FALSE;
    int   logLength;

    // Compile shader
    std::cout << "Compiling " << shaderType << " shader." << std::endl;
    glShaderSource(shader, 1, &shaderSrc, NULL);
    glCompileShader(shader);

    // Check shader
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> shaderError((logLength > 1) ? logLength : 1);
    glGetShaderInfoLog(shader, logLength, NULL, &shaderError[0]);
    std::cout << &shaderError[0] << std::endl;
    return shader;
}

glm::mat4 transform, View, Proj;
glm::vec3 centerEsfer, up;
GLuint    transViewProjLoc, radLoc;
float     ra = float(W / float(H)), radi;

float zoom        = 1.0f, scale = 1.0f;
int   rotate      = 0;
float angleY      = 0.0f, angleZ = 0.0f, angleX = 0.0f;
bool  perspectiva = false;
float maxCube;

void calcEsfera(glm::vec3 mins, glm::vec3 maxs, GLfloat bigCubeRad) {
    if (mins.x == maxs.x and mins.y == maxs.y and mins.z == maxs.z) {
        float minx, miny, minz, maxx, maxy, maxz;
        maxCube = maxz = maxy = maxx =  bigCubeRad * 0.4f; // experimental
                                                           // number

        minz = miny = minx = -maxCube;
        float zs = (maxz - minz) / 2.0f;
        float ys = (maxy - miny) / 2.0f;
        float xs = (maxx - minx) / 2.0f;
        centerEsfer = glm::vec3(0.0, 0.0, 0.0);
        radi        = xs;
        if (ys > radi) radi = ys;
        if (zs > radi) radi = zs;
    }
    else {
        float zs = (maxs.z - mins.z) / 2.0f;
        float ys = (maxs.y - mins.y) / 2.0f;
        float xs = (maxs.x - mins.x) / 2.0f;
        centerEsfer = glm::vec3(xs + mins.x, ys + mins.y, zs + mins.z);
        radi        = xs;
        if (ys > radi) radi = ys;
        if (zs > radi) radi = zs;
    }
}

void modelTransform() {
    transform = glm::mat4(1.0f);
    transform = glm::scale(transform, glm::vec3(scale));
    transform = glm::rotate(transform, (3.141516f / 4.0f) * rotate,
                            glm::vec3(0.0f, 1.0f, 0.0f));
}

void viewTransform() {
    View = glm::lookAt(centerEsfer + (radi * glm::vec3(0, 0, 1)),
                       centerEsfer, up);
    View = glm::rotate(View, -angleY, glm::vec3(0, 1, 0));
    View = glm::rotate(View, -angleZ, glm::vec3(0, 0, 1));
    View = glm::rotate(View, -angleX, glm::vec3(1, 0, 0));
}

void projecTransform() {
    if (perspectiva) Proj = glm::perspective(((float)3.141516f / 2.0f) * zoom, ra, 1.0f, maxCube * 3.0f);
    else Proj = glm::ortho(-radi * ra, radi * ra, -radi, radi, 0.01f, maxCube * 3.0f);
}

void updateViewProjectMatrix() {
    glm::mat4 transViewProj = Proj * View * transform;
    glUniformMatrix4fv(transViewProjLoc, 1, GL_FALSE, &transViewProj[0][0]);
}

void setRadius(GLfloat rad) {
    glUniform1f(radLoc, rad);
}

void loadUniforms(GLuint program, GLfloat cubeRad, GLfloat bigCubeRad) {
    calcEsfera(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), bigCubeRad);
    up =  glm::vec3(0, 1, 0);
    projecTransform();
    viewTransform();
    modelTransform();

    transViewProjLoc = glGetUniformLocation(program, "transViewProectionMatrix");
    radLoc           = glGetUniformLocation(program, "r");
    updateViewProjectMatrix();
    setRadius(cubeRad);
}


GLuint LoadShader(const char *geometryShader, const char *vertexShader, const char *fragmentShader) {
    GLuint geomShader = procesShaderFile(geometryShader, GL_GEOMETRY_SHADER);
    GLuint fragShader = procesShaderFile(fragmentShader, GL_FRAGMENT_SHADER);
    GLuint vertShader = procesShaderFile(vertexShader, GL_VERTEX_SHADER);

    std::cout << "Linking program." << std::endl;
    GLuint program = glCreateProgram();

    glAttachShader(program, geomShader);
    glAttachShader(program, fragShader);
    glAttachShader(program, vertShader);
    glLinkProgram(program);

    GLint result = GL_FALSE;
    int   logLength;
    glGetProgramiv(program, GL_LINK_STATUS, &result);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> programError((logLength > 1) ? logLength : 1);
    glGetProgramInfoLog(program, logLength, NULL, &programError[0]);
    std::cout << &programError[0] << std::endl;

    glDeleteShader(geomShader);
    glDeleteShader(fragShader);
    glDeleteShader(vertShader);
    return program;
}
