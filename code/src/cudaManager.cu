#include "../header/cudaManager.h"

__global__ void simple_vbo_kernel(float4 *pos, float4 *vel, unsigned int numCubesX, unsigned int numCubesY, unsigned int numCubesZ, float cubeSize) {
    // calculate index
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // calculate uv coordinates
    float offSet = (float)(1.0 / 2.0 * cubeSize);
    float u      = ((x / (float)numCubesX) * cubeSize - offSet);
    float w      = ((y / (float)numCubesY) * cubeSize - offSet);
    float v      = ((z / (float)numCubesZ) * cubeSize - offSet);

    // write output vertex
    pos[x + numCubesY * (y + numCubesZ * z)] = make_float4(u, w, v, 1.0f);
    vel[x + numCubesY * (y + numCubesZ * z)] = make_float4(0, 0, 0, 1.0f);
}


void initCubesDataKernal(float4 *ptr_pos, float4 *ptr_vel, unsigned int numCubesX, unsigned int numCubesY, unsigned int numCubesZ, float cubeSize) {
    dim3 block(8, 8, 8);
    dim3 grid(numCubesX / block.x, numCubesY / block.y, numCubesZ / block.z);
    simple_vbo_kernel << < grid, block >> > (ptr_pos, ptr_vel, numCubesX, numCubesY, numCubesZ, cubeSize);
}
void cubesUpdate(float4 *ptr_pos, float4 *ptr_vel, unsigned int numCubes) {}
