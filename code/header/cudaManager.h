#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H

struct float4;
void vertexKernelLauncher(float4 *pos, unsigned int numCubesX, unsigned int numCubesY, unsigned int numCubesZ, float deltaTime);

#endif // ifndef CUDAMANAGER_H
