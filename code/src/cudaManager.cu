#include "../header/cudaManager.h"

__global__ void calculate_vel_and_pos(float4 *pos, float4 *vel, unsigned int numCubesX, unsigned int numCubesY, unsigned int numCubesZ, float cubeSize) {
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
    float lenght   = sqrt(u * u + w * w + v * v);
    float totalVel = 0.1f;
    float u_vel    = (u / lenght) * totalVel;
    float w_vel    = (w / lenght) * totalVel;
    float v_vel    = (v / lenght) * totalVel;
    vel[x + numCubesY * (y + numCubesZ * z)] = make_float4(u_vel, w_vel, v_vel, 1.0f);
}


void initCubesDataKernal(float4 *ptr_pos, float4 *ptr_vel, unsigned int numCubesX, unsigned int numCubesY, unsigned int numCubesZ, float cubeSize) {
    dim3 block(8, 8, 8);
    dim3 grid(numCubesX / block.x, numCubesY / block.y, numCubesZ / block.z);
    calculate_vel_and_pos << < grid, block >> > (ptr_pos, ptr_vel, numCubesX, numCubesY, numCubesZ, cubeSize);
}



__global__ void calculate_update(float4 *pos, float4 *vel, float deltaTime, unsigned int numCubesY, unsigned int numCubesZ) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    float4 oldPos    = pos[x + numCubesY * (y + numCubesZ * z)];
    float4 actualVel = vel[x + numCubesY * (y + numCubesZ * z)];
    float  u         = oldPos.x + deltaTime * actualVel.x;
    float  w         = oldPos.y + deltaTime * actualVel.y;
    float  v         = oldPos.z + deltaTime * actualVel.z;

    // write output vertex
    pos[x + numCubesY * (y + numCubesZ * z)] = make_float4(u, w, v, 1.0f);
}

void cubesUpdate(float4 *ptr_pos, float4 *ptr_vel, unsigned int numCubesX, unsigned int numCubesY, unsigned int numCubesZ, float deltaTime) {
    dim3 block(8, 8, 8);
    dim3 grid(numCubesX / block.x, numCubesY / block.y, numCubesZ / block.z);
    calculate_update << < grid, block >> > (ptr_pos, ptr_vel, deltaTime, numCubesY, numCubesZ);
}
