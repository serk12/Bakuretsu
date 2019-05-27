#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H

struct float4;
void initCubesDataKernal(float4 *ptr_pos, float4 *ptr_vel, unsigned int numCubesX, unsigned int numCubesY, unsigned int numCubesZ, float cubeSize);
void cubesUpdate(float4 *ptr_pos, float4 *ptr_vel, unsigned int numCubesX, unsigned int numCubesY, unsigned int numCubesZ, float bigCubeRad, float deltaTime);
#endif // ifndef CUDAMANAGER_H
