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

glm::vec3 centerEsfer, up;
GLuint    transLoc, projLoc, viewLoc;
float     ra = float(600 / float(600)), radi;

float zoom        = 1.0f, scale = 1.0f;
int   rotate      = 0;
float angleY      = 0.0f, angleZ = 0.0f, angleX = 0.0f;
bool  perspectiva = false;

void calcEsfera(glm::vec3 mins, glm::vec3 maxs) {
    if (mins.x == maxs.x and mins.y == maxs.y and mins.z == maxs.z) {
        float minx, miny, minz, maxx, maxy, maxz;
        maxz = maxy = maxx = 8 * 1.7f / 2.0f; // alert hardcoded numCubesX and
                                              // cubeSize
        minz = miny = minx = -maxz;
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
    glm::mat4 transform(1.0f);
    transform = glm::scale(transform, glm::vec3(scale));
    transform = glm::rotate(transform, (3.141516f / 4.0f) * rotate,
                            glm::vec3(0.0f, 1.0f, 0.0f));
    glUniformMatrix4fv(transLoc, 1, GL_FALSE, &transform[0][0]);
}

void viewTransform() {
    glm::mat4 View = glm::lookAt(centerEsfer + (radi * glm::vec3(0, 0, 1)),
                                 centerEsfer, up);
    View = glm::rotate(View, -angleY, glm::vec3(0, 1, 0));
    View = glm::rotate(View, -angleZ, glm::vec3(0, 0, 1));
    View = glm::rotate(View, -angleX, glm::vec3(1, 0, 0));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &View[0][0]);
}

void projecTransform() {
    glm::mat4 Proj;
    if (perspectiva) Proj = glm::perspective(((float)3.141516f / 2.0f) * zoom, ra, 0.01f, 20.0f);
    else Proj = glm::ortho(-radi * ra, radi * ra, -radi, radi, 0.01f, 20.0f);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, &Proj[0][0]);
}

void loadUniforms(GLuint program) {
    transLoc = glGetUniformLocation(program, "TG");
    projLoc  = glGetUniformLocation(program, "proj");
    viewLoc  = glGetUniformLocation(program, "view");

    calcEsfera(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0));
    up =  glm::vec3(0, 1, 0);
    projecTransform();
    viewTransform();
    modelTransform();
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