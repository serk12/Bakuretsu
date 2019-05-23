#include "../header/cudaManager.h"

void vertexKernelLauncher(float4 *pos, unsigned int numCubes, float deltaTime) {
    // dim3 block(8, 8, 1);
    // dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    // simple_vbo_kernel << < grid, block >> > (pos, mesh_width, mesh_height,
    // time);
}
