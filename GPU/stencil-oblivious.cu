#include "common.h"
#include <stdio.h>
const char* version_name = "cuda oblivious version";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    /* multi-threads in one machine */
    grid_info->local_size_x = grid_info->global_size_x;
    grid_info->local_size_y = grid_info->global_size_y;
    grid_info->local_size_z = grid_info->global_size_z;
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {
    
}

__global__ void stencil_7_naive_kernel_1step(cptr_t in, ptr_t out, \
    int nx, int ny, int nz, \
    int halo_x, int halo_y, int halo_z, int ldx, int ldy) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;
    if(tx < nx && ty < ny && tz < nz) {
        int x = tx + halo_x;
        int y = ty + halo_y;
        int z = tz + halo_z;
        out[INDEX(x, y, z, ldx, ldy)] \
            = ALPHA_ZZZ * in[INDEX(x, y, z, ldx, ldy)] \
            + ALPHA_NZZ * in[INDEX(x-1, y, z, ldx, ldy)] \
            + ALPHA_PZZ * in[INDEX(x+1, y, z, ldx, ldy)] \
            + ALPHA_ZNZ * in[INDEX(x, y-1, z, ldx, ldy)] \
            + ALPHA_ZPZ * in[INDEX(x, y+1, z, ldx, ldy)] \
            + ALPHA_ZZN * in[INDEX(x, y, z-1, ldx, ldy)] \
            + ALPHA_ZZP * in[INDEX(x, y, z+1, ldx, ldy)];
    }
}

__global__ void stencil_27_naive_kernel_1step(cptr_t in, ptr_t out, \
    int nx, int ny, int nz, \
    int halo_x, int halo_y, int halo_z, int ldx, int ldy) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;
    if(tx < nx && ty < ny && tz < nz) {
        int x = tx + halo_x;
        int y = ty + halo_y;
        int z = tz + halo_z;
        out[INDEX(x, y, z, ldx, ldy)] \
            = ALPHA_ZZZ * in[INDEX(x, y, z, ldx, ldy)] \
            + ALPHA_NZZ * in[INDEX(x-1, y, z, ldx, ldy)] \
            + ALPHA_PZZ * in[INDEX(x+1, y, z, ldx, ldy)] \
            + ALPHA_ZNZ * in[INDEX(x, y-1, z, ldx, ldy)] \
            + ALPHA_ZPZ * in[INDEX(x, y+1, z, ldx, ldy)] \
            + ALPHA_ZZN * in[INDEX(x, y, z-1, ldx, ldy)] \
            + ALPHA_ZZP * in[INDEX(x, y, z+1, ldx, ldy)] \
            + ALPHA_NNZ * in[INDEX(x-1, y-1, z, ldx, ldy)] \
            + ALPHA_PNZ * in[INDEX(x+1, y-1, z, ldx, ldy)] \
            + ALPHA_NPZ * in[INDEX(x-1, y+1, z, ldx, ldy)] \
            + ALPHA_PPZ * in[INDEX(x+1, y+1, z, ldx, ldy)] \
            + ALPHA_NZN * in[INDEX(x-1, y, z-1, ldx, ldy)] \
            + ALPHA_PZN * in[INDEX(x+1, y, z-1, ldx, ldy)] \
            + ALPHA_NZP * in[INDEX(x-1, y, z+1, ldx, ldy)] \
            + ALPHA_PZP * in[INDEX(x+1, y, z+1, ldx, ldy)] \
            + ALPHA_ZNN * in[INDEX(x, y-1, z-1, ldx, ldy)] \
            + ALPHA_ZPN * in[INDEX(x, y+1, z-1, ldx, ldy)] \
            + ALPHA_ZNP * in[INDEX(x, y-1, z+1, ldx, ldy)] \
            + ALPHA_ZPP * in[INDEX(x, y+1, z+1, ldx, ldy)] \
            + ALPHA_NNN * in[INDEX(x-1, y-1, z-1, ldx, ldy)] \
            + ALPHA_PNN * in[INDEX(x+1, y-1, z-1, ldx, ldy)] \
            + ALPHA_NPN * in[INDEX(x-1, y+1, z-1, ldx, ldy)] \
            + ALPHA_PPN * in[INDEX(x+1, y+1, z-1, ldx, ldy)] \
            + ALPHA_NNP * in[INDEX(x-1, y-1, z+1, ldx, ldy)] \
            + ALPHA_PNP * in[INDEX(x+1, y-1, z+1, ldx, ldy)] \
            + ALPHA_NPP * in[INDEX(x-1, y+1, z+1, ldx, ldy)] \
            + ALPHA_PPP * in[INDEX(x+1, y+1, z+1, ldx, ldy)];
    }
}
inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}
#define BLOCK_SIZE 8
void walk3(ptr_t buffer[], int nx, int ny, int nz, int t0, int t1,
            int type, int x0, int dx0, int x1, int dx1,
                      int y0, int dy0, int y1, int dy1,
                      int z0, int dz0, int z1, int dz1){
    int dt = t1 - t0;
    if(dt == 1 || (x1-x0)*(y1-y0)*(z1-z0) < 2097152){
        for(int t = t0; t < t1; t++){
            dim3 grid_size (ceiling((x1 - x0) + (t - t0) * (dx1 - dx0), BLOCK_SIZE),  
                            ceiling((y1 - y0) + (t - t0) * (dy1 - dy0), BLOCK_SIZE), 
                            ceiling((z1 - z0) + (t - t0) * (dz1 - dz0), BLOCK_SIZE));
            dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
            if(type == 7){
                stencil_7_naive_kernel_1step<<<grid_size, block_size>>>(\
                    buffer[t % 2], buffer[(t + 1) % 2], (x1 - x0) + (t - t0) * (dx1 - dx0),\
                        (y1 - y0) + (t - t0) * (dy1 - dy0), (z1 - z0) + (t - t0) * (dz1 - dz0), \
                        x0 + dx0 * (t - t0), y0 + dy0 * (t - t0),\
                        z0 + dz0 * (t - t0), nx, ny);
            }
            else{
                stencil_27_naive_kernel_1step<<<grid_size, block_size>>>(\
                    buffer[t % 2], buffer[(t + 1) % 2], (x1 - x0) + (t - t0) * (dx1 - dx0),\
                        (y1 - y0) + (t - t0) * (dy1 - dy0), (z1 - z0) + (t - t0) * (dz1 - dz0), \
                        x0 + dx0 * (t - t0), y0 + dy0 * (t - t0),\
                        z0 + dz0 * (t - t0), nx, ny);
            }
            cudaDeviceSynchronize();
        }
    }
    else if (dt > 1) {
    if (2 * (z1 - z0) + (dz1 - dz0) * dt >= 4 * dt) {
        int zm = (2 * (z0 + z1) + (2 + dz0 + dz1) * dt) / 4;
        walk3(buffer, nx, ny, nz, t0, t1, type, 
            x0, dx0, x1, dx1, 
            y0, dy0, y1, dy1,
            z0, dz0, zm, -1);
        walk3(buffer, nx, ny, nz, t0, t1, type, 
            x0, dx0, x1, dx1, 
            y0, dy0, y1, dy1,
            zm, -1, z1, dz1);
    }
    else if (2 * (y1 - y0) + (dy1 - dy0) * dt >= 4 * dt) {
        int ym = (2 * (y0 + y1) + (2 + dy0 + dy1) * dt) / 4;
        walk3(buffer, nx, ny, nz, t0, t1, type, 
            x0, dx0, x1, dx1, 
            y0, dy0, ym, -1,
            z0, dz0, z1, dz1);
        walk3(buffer, nx, ny, nz, t0, t1, type, 
            x0, dx0, x1, dx1,
            ym, -1, y1, dy1,
            z0, dz0, z1, dz1);
    }
    else {
      int s = dt / 2;
      walk3(buffer, nx, ny, nz, t0, t0 + s, type, 
        x0, dx0, x1, dx1,
        y0, dy0, y1, dy1,
        z0, dz0, z1, dz1);
      walk3(buffer, nx, ny, nz, t0 + s, t1, type, 
        x0 + dx0 * s, dx0, x1 + dx1 * s, dx1, 
        y0 + dy0 * s, dy0, y1 + dy1 * s, dy1,
	    z0 + dz0 * s, dz0, z1 + dz1 * s, dz1);
    }
  }
}

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    walk3(buffer, ldx, ldy, ldz, 0, nt, 7,
        x_start, 0, x_end, 0,
        y_start, 0, y_end, 0,
        z_start, 0, z_end, 0);
    return buffer[nt % 2];
}

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    walk3(buffer, ldx, ldy, ldz, 0, nt, 27,
        x_start, 0, x_end, 0,
        y_start, 0, y_end, 0,
        z_start, 0, z_end, 0);
    return buffer[nt % 2];
}