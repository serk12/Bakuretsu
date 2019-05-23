#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H

struct float4;
void vertexKernelLauncher(float4 *pos, unsigned int numCubes, float deltaTime);

#endif // ifndef CUDAMANAGER_H
