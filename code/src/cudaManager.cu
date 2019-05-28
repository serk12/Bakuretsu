#include "../header/cudaManager.h"

const unsigned int numCubesX = 20;
const unsigned int numCubesY = 20;
const unsigned int numCubesZ = 20;
const unsigned int numCubes  = numCubesX * numCubesY * numCubesZ;
const float cubeSize         = 1.0f;
const float cubeDistance     = cubeSize + 0.01f;
// restitution
const float e = 0.50;
// invMass=1/(densiti*vol)
const float invMass = 1.0f / (0.6f * cubeSize * cubeSize * cubeSize);
const float initVel = 1.70f;

__global__ void calculate_vel_and_pos(float4 *pos, float4 *vel) {
    // calculate index
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // calculate uv coordinates
    float u = ((x * cubeDistance) - (cubeDistance * numCubesX / 2.0));
    float w = ((y * cubeDistance) - (cubeDistance * numCubesY / 2.0));
    float v = ((z * cubeDistance) - (cubeDistance * numCubesZ / 2.0));

    // write output vertex
    unsigned int i = x + numCubesY * (y + numCubesZ * z);
    pos[i] = make_float4(u, w, v, 1.0f);
    float lenght = sqrt(u * u + w * w + v * v);
    float u_vel  = (u / lenght) * initVel;
    float w_vel  = (w / lenght) * initVel;
    float v_vel  = (v / lenght) * initVel;
    vel[i] = make_float4(u_vel, w_vel, v_vel, 1.0f);
}


void initCubesDataKernal(float4 *ptr_pos, float4 *ptr_vel) {
    dim3 block(5, 5, 5);
    dim3 grid(numCubesX / block.x, numCubesY / block.y, numCubesZ / block.z);
    calculate_vel_and_pos << < grid, block >> > (ptr_pos, ptr_vel);
}



__global__ void calculate_update(float4 *pos, float4 *vel, float deltaTime, float bigCubeRad) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float  dx = deltaTime * vel[i].x;
    float  dy = deltaTime * vel[i].y;
    float  dz = deltaTime * vel[i].z;
    float4 a  = pos[i];
    if ((a.x + dx > bigCubeRad - cubeSize) || (a.x + dx < -bigCubeRad)) {
        vel[i].x = -vel[i].x;
    }
    if ((a.y + dy > bigCubeRad - cubeSize) || (a.y + dy < -bigCubeRad)) {
        vel[i].y = -vel[i].y;
    }
    if ((a.z + dz > bigCubeRad - cubeSize) || (a.z + dz < -bigCubeRad)) {
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
        if (((a.x <= b.x + cubeSize) && (a.x + cubeSize >= b.x)) &&
            ((a.y <= b.y + cubeSize) && (a.y + cubeSize >= b.y)) &&
            ((a.z <= b.z + cubeSize) && (a.z + cubeSize >= b.z))) {
            float  directionX = pos[j].x - pos[i].x;
            float  directionY = pos[j].y - pos[i].y;
            float  directionZ = pos[j].z - pos[i].z;
            float  len        = sqrt(directionX * directionX + directionY * directionY + directionZ * directionZ);
            float4 n          = make_float4(directionX / len, directionY / len, directionZ / len, 1.0f);

            float4 rv = make_float4(vel[j].x - vel[i].x, vel[j].y - vel[i].y, vel[j].z - vel[i].z, 1.0f);
            // // Calculate relative velocity in terms of the normal direction
            float velAlongNormal = rv.x * n.x + rv.y * n.y + rv.z * n.z;
            //
            // // Do not resolve if velocities are separating
            if (velAlongNormal > 0) return;

            // Calculate impulse scalar
            float k = (-(1 + e) * velAlongNormal) / invMass;
            // Apply impulse
            float4 impulse = make_float4(k * n.x, k * n.y, k * n.z, 1.0f);
            vel[i] = make_float4(vel[i].x - impulse.x, vel[i].y - impulse.y, vel[i].z - impulse.z, 1.0f);
            vel[j] = make_float4(vel[j].x + impulse.x, vel[j].y + impulse.y, vel[j].z + impulse.z, 1.0f);
        }
    }
}


void cubesUpdate(float4 *ptr_pos, float4 *ptr_vel, float bigCubeRad, float deltaTime) {
    // unsigned int triangleNumberN = numCubes - 1;
    // diagonal + 1 / 2 = diag^2+diag / (2*diag) [+1 for ceiling]
    // unsigned int dimToScale = (triangleNumberN + 2) / 2;
    dim3 block(32, 32);
    dim3 grid(numCubes / block.x, numCubes / block.y);
    calculate_collision << < grid, block  >> > (ptr_pos, ptr_vel, deltaTime);
    calculate_update << < numCubes / 64, 64 >> > (ptr_pos, ptr_vel, deltaTime, bigCubeRad / 2.0f);
}
