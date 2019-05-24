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
