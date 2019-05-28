#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H

#include <iostream>

struct float4;
extern const unsigned int numCubesX, numCubesY, numCubesZ, numCubes;
extern const float cubeDistance, cubeSize;

void initCubesDataKernal(float4 *ptr_pos, float4 *ptr_vel);
void cubesUpdate(float4 *ptr_pos, float4 *ptr_vel, float bigCubeRad, float deltaTime);
#endif // ifndef CUDAMANAGER_H
