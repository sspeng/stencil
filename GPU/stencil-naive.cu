#include "common.h"
#include <stdio.h>
#define XBS 32
#define YBS 4
#define ZBS 4
const int XX = XBS + 2;
const int YY = YBS + 2;
const int ZZ = ZBS + 2;
const char* version_name = "A naive base-line";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

__global__ void stencil_7_naive_kernel_1step(cptr_t in, ptr_t out, \
                                int nx, int ny, int nz, \
                                int halo_x, int halo_y, int halo_z) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;
    if(tx < nx && ty < ny && tz < nz) {
        int ldx = nx + halo_x * 2;
        int ldy = ny + halo_y * 2;
        int x = tx + halo_x;
        int y = ty + halo_y;
        int z = tz + halo_z;

        __shared__ double s_in[XX * YY * ZZ];

        int xx = threadIdx.x + 1;
        int yy = threadIdx.y + 1;
        int zz = threadIdx.z + 1;

        int base1 = xx + XX * (yy + YY * zz);
        int base2 = x + ldx * (y + ldy * z);
        int bz1 = XX * YY;
        int bz2 = ldx * ldy;

        s_in[base1] = in[base2];
        if(xx == 1)   s_in[base1 - 1] = in[base2 - 1];
        if(xx == XBS) s_in[base1 + 1] = in[base2 + 1];
        if(yy == 1)   s_in[base1 - XX] = in[base2 - ldx];
        if(yy == YBS) s_in[base1 + XX] = in[base2 + ldx];
        if(zz == 1)   s_in[base1 - bz1] = in[base2 - bz2];
        if(zz == ZBS) s_in[base1 + bz1] = in[base2 + bz2];

        __syncthreads();

        out[base2] \
            = ALPHA_ZZZ * s_in[base1] \
            + ALPHA_NZZ * s_in[base1 - 1] \
            + ALPHA_PZZ * s_in[base1 + 1] \
            + ALPHA_ZNZ * s_in[base1 - XX] \
            + ALPHA_ZPZ * s_in[base1 + XX] \
            + ALPHA_ZZN * s_in[base1 - bz1] \
            + ALPHA_ZZP * s_in[base1 + bz1];
    }
}

__global__ void stencil_27_naive_kernel_1step(cptr_t in, ptr_t out, \
                                int nx, int ny, int nz, \
                                int halo_x, int halo_y, int halo_z) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;
    if(tx < nx && ty < ny && tz < nz) {
        int ldx = nx + halo_x * 2;
        int ldy = ny + halo_y * 2;
        int x = tx + halo_x;
        int y = ty + halo_y;
        int z = tz + halo_z;

        __shared__ double s_in[ZZ][YY][XX];

        int xx = threadIdx.x + 1;
        int yy = threadIdx.y + 1;
        int zz = threadIdx.z + 1;

        int dx, dy, dz;
        dx = dy = dz = 0;
        if(xx == 1)   dx = -1;
        if(xx == XBS) dx = 1;
        if(yy == 1)   dy = -1;
        if(yy == YBS) dy = 1;
        if(zz == 1)   dz = -1;
        if(zz == ZBS) dz = 1;

        s_in[zz][yy][xx] = in[INDEX(x, y, z, ldx, ldy)];
        if(dx) s_in[zz][yy][xx + dx] = in[INDEX(x + dx, y, z, ldx, ldy)];
        if(dy) s_in[zz][yy + dy][xx] = in[INDEX(x, y + dy, z, ldx, ldy)];
        if(dz) s_in[zz + dz][yy][xx] = in[INDEX(x, y, z + dz, ldx, ldy)];
        if(dx && dy) s_in[zz][yy + dy][xx + dx] = in[INDEX(x + dx, y + dy, z, ldx, ldy)];
        if(dx && dz) s_in[zz + dz][yy][xx + dx] = in[INDEX(x + dx, y, z + dz, ldx, ldy)];
        if(dy && dz) s_in[zz + dz][yy + dy][xx] = in[INDEX(x, y + dy, z + dz, ldx, ldy)];
        if(dx && dy && dz) s_in[zz + dz][yy + dy][xx + dx] = in[INDEX(x + dx, y + dy, z + dz, ldx, ldy)];
        // int minx = (dx < 0) ? dx : 0;
        // int maxx = (dx < 0) ? 0 : dx;
        // int miny = (dy < 0) ? dy : 0;
        // int maxy = (dy < 0) ? 0 : dy;
        // int minz = (dz < 0) ? dz : 0;
        // int maxz = (dz < 0) ? 0 : dz;
        // for(int i = minz; i <= maxz; i++){
        //     for(int j = miny; j <= maxy; j++){
        //         for(int k = minx; k <= maxx; k++){
        //             s_in[zz + i][yy + j][xx + k] = in[INDEX(x + k, y + j, z + i, ldx, ldy)];
        //         }
        //     }
        // }

        __syncthreads();

        out[INDEX(x, y, z, ldx, ldy)] \
            = ALPHA_ZZZ * s_in[zz][yy][xx] \
            + ALPHA_NZZ * s_in[zz][yy][xx - 1] \
            + ALPHA_PZZ * s_in[zz][yy][xx + 1] \
            + ALPHA_ZNZ * s_in[zz][yy - 1][xx] \
            + ALPHA_ZPZ * s_in[zz][yy + 1][xx] \
            + ALPHA_ZZN * s_in[zz - 1][yy][xx] \
            + ALPHA_ZZP * s_in[zz + 1][yy][xx] \
            + ALPHA_NNZ * s_in[zz][yy - 1][xx - 1] \
            + ALPHA_PNZ * s_in[zz][yy - 1][xx + 1] \
            + ALPHA_NPZ * s_in[zz][yy + 1][xx - 1] \
            + ALPHA_PPZ * s_in[zz][yy + 1][xx + 1] \
            + ALPHA_NZN * s_in[zz - 1][yy][xx - 1] \
            + ALPHA_PZN * s_in[zz - 1][yy][xx + 1] \
            + ALPHA_NZP * s_in[zz + 1][yy][xx - 1] \
            + ALPHA_PZP * s_in[zz + 1][yy][xx + 1] \
            + ALPHA_ZNN * s_in[zz - 1][yy - 1][xx] \
            + ALPHA_ZPN * s_in[zz - 1][yy + 1][xx] \
            + ALPHA_ZNP * s_in[zz + 1][yy - 1][xx] \
            + ALPHA_ZPP * s_in[zz + 1][yy + 1][xx] \
            + ALPHA_NNN * s_in[zz - 1][yy - 1][xx - 1] \
            + ALPHA_PNN * s_in[zz - 1][yy - 1][xx + 1] \
            + ALPHA_NPN * s_in[zz - 1][yy + 1][xx - 1] \
            + ALPHA_PPN * s_in[zz - 1][yy + 1][xx + 1] \
            + ALPHA_NNP * s_in[zz + 1][yy - 1][xx - 1] \
            + ALPHA_PNP * s_in[zz + 1][yy - 1][xx + 1] \
            + ALPHA_NPP * s_in[zz + 1][yy + 1][xx - 1] \
            + ALPHA_PPP * s_in[zz + 1][yy + 1][xx + 1];
    }
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    // cudaDeviceProp devp;
    // int dev = 0;
    // cudaGetDeviceProperties(&devp, dev);
    // printf("%d\n",  devp.maxThreadsPerBlock);
    dim3 grid_size (ceiling(nx, XBS), ceiling(ny, YBS), ceiling(nz, ZBS));
    dim3 block_size (XBS, YBS, ZBS);
    for(int t = 0; t < nt; ++t) {
        stencil_7_naive_kernel_1step<<<grid_size, block_size>>>(\
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
                grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int nx = grid_info->global_size_x;
    int ny = grid_info->global_size_y;
    int nz = grid_info->global_size_z;
    dim3 grid_size (ceiling(nx, XBS), ceiling(ny, YBS), ceiling(nz, ZBS));
    dim3 block_size (XBS, YBS, ZBS);
    for(int t = 0; t < nt; ++t) {
        stencil_27_naive_kernel_1step<<<grid_size, block_size>>>(\
            buffer[t % 2], buffer[(t + 1) % 2], nx, ny, nz, \
                grid_info->halo_size_x, grid_info->halo_size_y, grid_info->halo_size_z);
    }
    return buffer[nt % 2];
}