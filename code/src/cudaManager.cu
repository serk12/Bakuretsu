#include "../header/cudaManager.h"

const unsigned int numCubesX = 8;
const unsigned int numCubesY = 8;
const unsigned int numCubesZ = 8;
const unsigned int numCubes  = numCubesX * numCubesY * numCubesZ;
const float cubeSize         = numCubesY + 0.3f;

__global__ void calculate_vel_and_pos(float4 *pos, float4 *vel) {
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
    float totalVel = 1.1f;
    float u_vel    = (u / lenght) * totalVel;
    float w_vel    = (w / lenght) * totalVel;
    float v_vel    = (v / lenght) * totalVel;
    vel[x + numCubesY * (y + numCubesZ * z)] = make_float4(u_vel, w_vel, v_vel, 1.0f);
}


void initCubesDataKernal(float4 *ptr_pos, float4 *ptr_vel) {
    dim3 block(8, 8, 8);
    dim3 grid(numCubesX / block.x, numCubesY / block.y, numCubesZ / block.z);
    calculate_vel_and_pos << < grid, block >> > (ptr_pos, ptr_vel);
}



__global__ void calculate_update(float4 *pos, float4 *vel, float deltaTime, float bigCubeRad) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    float4 oldPos  = pos[x + numCubesY * (y + numCubesZ * z)];
    if ((oldPos.x > bigCubeRad) || (oldPos.x < -bigCubeRad)) {
        vel[x + numCubesY * (y + numCubesZ * z)].x = -vel[x + numCubesY * (y + numCubesZ * z)].x;
    }
    if ((oldPos.y > bigCubeRad) || (oldPos.y < -bigCubeRad)) {
        vel[x + numCubesY * (y + numCubesZ * z)].y = -vel[x + numCubesY * (y + numCubesZ * z)].y;
    }
    if ((oldPos.z > bigCubeRad) || (oldPos.z < -bigCubeRad)) {
        vel[x + numCubesY * (y + numCubesZ * z)].z = -vel[x + numCubesY * (y + numCubesZ * z)].z;
    }


    float4 actualVel = vel[x + numCubesY * (y + numCubesZ * z)];
    float  u         = oldPos.x + deltaTime * actualVel.x;
    float  w         = oldPos.y + deltaTime * actualVel.y;
    float  v         = oldPos.z + deltaTime * actualVel.z;

    pos[x + numCubesY * (y + numCubesZ * z)] = make_float4(u, w, v, 1.0f);
}

void cubesUpdate(float4 *ptr_pos, float4 *ptr_vel, float bigCubeRad, float deltaTime) {
    dim3 block(8, 8, 8);
    dim3 grid(numCubesX / block.x, numCubesY / block.y, numCubesZ / block.z);
    calculate_update << < grid, block >> > (ptr_pos, ptr_vel, deltaTime, bigCubeRad / 2.0f);
}
