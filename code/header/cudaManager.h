#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H

struct float4;
extern const unsigned int numCubesX, numCubesY, numCubesZ, numCubes;
extern const float cubeSize;

void initCubesDataKernal(float4 *ptr_pos, float4 *ptr_vel);
void cubesUpdate(float4 *ptr_pos, float4 *ptr_vel, float bigCubeRad, float deltaTime);
#endif // ifndef CUDAMANAGER_H
