#include "../header/cudaManager.h"

const unsigned int numCubesX = 8;
const unsigned int numCubesY = 8;
const unsigned int numCubesZ = 8;
const unsigned int numCubes  = numCubesX * numCubesY * numCubesZ;
const float cubeSize         = (numCubesY > numCubesX ?
                                (numCubesY > numCubesZ ? numCubesY : numCubesZ) :
                                (numCubesX > numCubesZ ? numCubesX : numCubesZ)) + 0.1f;

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
    unsigned int i = x + numCubesY * (y + numCubesZ * z);
    pos[i] = make_float4(u, w, v, 1.0f);
    float lenght   = sqrt(u * u + w * w + v * v);
    float totalVel = 1.1f;
    float u_vel    = (u / lenght) * totalVel;
    float w_vel    = (w / lenght) * totalVel;
    float v_vel    = (v / lenght) * totalVel;
    vel[i] = make_float4(u_vel, w_vel, v_vel, 1.0f);
}


void initCubesDataKernal(float4 *ptr_pos, float4 *ptr_vel) {
    dim3 block(1, 1, 1);
    dim3 grid(numCubesX / block.x, numCubesY / block.y, numCubesZ / block.z);
    calculate_vel_and_pos << < grid, block >> > (ptr_pos, ptr_vel);
}



__global__ void calculate_update(float4 *pos, float4 *vel, float deltaTime, float bigCubeRad) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float  dx = deltaTime * vel[i].x;
    float  dy = deltaTime * vel[i].y;
    float  dz = deltaTime * vel[i].z;
    float4 a  = pos[i];
    if ((a.x + dx > bigCubeRad + cubeSize) || (a.x + dx < -bigCubeRad)) {
        vel[i].x = -vel[i].x;
    }
    if ((a.y + dy > bigCubeRad + cubeSize) || (a.y + dy < -bigCubeRad)) {
        vel[i].y = -vel[i].y;
    }
    if ((a.z + dz > bigCubeRad + cubeSize) || (a.z + dz < -bigCubeRad)) {
        vel[i].z = -vel[i].z;
    }

    float u = pos[i].x + deltaTime * vel[i].x;
    float w = pos[i].y + deltaTime * vel[i].y;
    float v = pos[i].z + deltaTime * vel[i].z;
    pos[i] = make_float4(u, w, v, 1.0f);
}

__global__ void calculate_collision(float4 *pos, float4 *vel, float deltaTime) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < j) {
        float4 a = pos[i];
        float4 b = pos[j];
        if (((a.x <= b.x + 1.0f) && (a.x + 1.0f >= b.x)) &&
            ((a.y <= b.y + 1.0f) && (a.y + 1.0f >= b.y)) &&
            ((a.z <= b.z + 1.0f) && (a.z + 1.0f >= b.z))) {
            vel[i].x = -vel[i].x;
            vel[i].y = -vel[i].y;
            vel[i].z = -vel[i].z;
        }
    }
}


void cubesUpdate(float4 *ptr_pos, float4 *ptr_vel, float bigCubeRad, float deltaTime) {
    // unsigned int triangleNumberN = numCubes - 1;
    // diagonal + 1 / 2 = diag^2+diag / (2*diag) [+1 for ceiling]
    // unsigned int dimToScale = (triangleNumberN + 2) / 2;
    dim3 block(8, 8);
    dim3 grid(numCubes / 8, numCubes / 8);
    calculate_collision << < grid, block >> > (ptr_pos, ptr_vel, deltaTime);
    calculate_update << < 8, numCubes / 8 >> > (ptr_pos, ptr_vel, deltaTime, bigCubeRad / 2.0f);
}
