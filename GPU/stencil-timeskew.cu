#include "common.h"
#include <stdio.h>

#define BLOCK_SIZE 64
#define max(a, b) (a > b) ? a : b
#define XBS 8
#define YBS 8
#define ZBS 8
const char* version_name = "timeskew version";

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
        double sum = 0;
        int x = tx + halo_x;
        int y = ty + halo_y;
        int z = tz + halo_z;
        int xx = threadIdx.x + 1;
        int yy = threadIdx.y + 1;
        int zz = threadIdx.z + 1;
        __shared__ double s_in[XBS + 2][YBS + 2][ZBS + 2];
        s_in[xx][yy][zz] = in[INDEX(x, y, z, ldx, ldy)];
        if(xx == 1)   s_in[xx - 1][yy][zz] = in[INDEX(x - 1, y, z, ldx, ldy)];
        if(xx == XBS) s_in[xx + 1][yy][zz] = in[INDEX(x + 1, y, z, ldx, ldy)];
        if(yy == 1)   s_in[xx][yy - 1][zz] = in[INDEX(x, y - 1, z, ldx, ldy)];
        if(yy == YBS) s_in[xx][yy + 1][zz] = in[INDEX(x, y + 1, z, ldx, ldy)];
        if(zz == 1)   s_in[xx][yy][zz - 1] = in[INDEX(x, y, z - 1, ldx, ldy)];
        if(zz == ZBS) s_in[xx][yy][zz + 1] = in[INDEX(x, y, z + 1, ldx, ldy)];
        __syncthreads();

        sum = ALPHA_ZZZ * s_in[xx][yy][zz] \
            + ALPHA_NZZ * s_in[xx - 1][yy][zz] \
            + ALPHA_PZZ * s_in[xx + 1][yy][zz] \
            + ALPHA_ZNZ * s_in[xx][yy - 1][zz] \
            + ALPHA_ZPZ * s_in[xx][yy + 1][zz] \
            + ALPHA_ZZN * s_in[xx][yy][zz - 1] \
            + ALPHA_ZZP * s_in[xx][yy][zz + 1];
        out[INDEX(x, y, z, ldx, ldy)] = sum;
    }
}

__global__ void stencil_27_naive_kernel_1step(cptr_t in, ptr_t out, \
    int nx, int ny, int nz, \
    int halo_x, int halo_y, int halo_z, int ldx, int ldy) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;
    
    while(tx < nx && ty < ny && tz < nz) {
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
        tx += blockDim.x * gridDim.x;
    }
}
inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    //int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    int bs_z = 128;
    for(int zz = z_start; zz < z_end; zz += bs_z){
        int neg_z = 1;
        int pos_z = -1;
        if(zz == z_start) neg_z = 0;
        if(zz == z_end - bs_z) pos_z = 0;

        int bs_y = 128;
        for(int yy = y_start; yy < y_end; yy += bs_y){
            int neg_y = 1;
            int pos_y = -1;
            if(yy == y_start) neg_y = 0;
            if(yy == y_end - bs_y) pos_y = 0;

            for(int xx = x_start; xx < x_end; xx += (x_end - x_start)){
                int neg_x = 1;
                int pos_x = -1;
                if(xx == x_start) neg_x = 0;
                if(xx == x_end - (x_end - x_start)) pos_x = 0;

                for(int t = 0; t < nt; t++){

                    int blockmin_x = max(x_start, xx - t * neg_x);
                    int blockmin_y = max(y_start, yy - t * neg_y);
                    int blockmin_z = max(z_start, zz - t * neg_z);

                    int blockmax_x = max(x_start, xx + (x_end-x_start) + t * pos_x);
                    int blockmax_y = max(y_start, yy + bs_y + t * pos_y);
                    int blockmax_z = max(z_start, zz + bs_z + t * pos_z);

                    dim3 grid_size (ceiling((blockmax_x - blockmin_x), XBS),  
                                    ceiling((blockmax_y - blockmin_y), YBS), 
                                    ceiling((blockmax_z - blockmin_z), ZBS));
                    dim3 block_size(XBS, YBS, ZBS);
                    stencil_7_naive_kernel_1step<<<grid_size, block_size>>>(\
                        buffer[t % 2], buffer[(t + 1) % 2], blockmax_x - blockmin_x,\
                            blockmax_y - blockmin_y, blockmax_z - blockmin_z, \
                            blockmin_x, blockmin_y,\
                            blockmin_z, ldx, ldy);
                }
            }
        }
    }
    return buffer[nt % 2];
}

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    //int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;
    
    for(int zz = z_start; zz < z_end; zz += BLOCK_SIZE){
        int neg_z = 1;
        int pos_z = -1;
        if(zz == z_start) neg_z = 0;
        if(zz == z_end - BLOCK_SIZE) pos_z = 0;

        for(int yy = y_start; yy < y_end; yy += BLOCK_SIZE){
            int neg_y = 1;
            int pos_y = -1;
            if(yy == y_start) neg_y = 0;
            if(yy == y_end - BLOCK_SIZE) pos_y = 0;

            for(int xx = x_start; xx < x_end; xx += BLOCK_SIZE){
                int neg_x = 1;
                int pos_x = -1;
                if(xx == x_start) neg_x = 0;
                if(xx == x_end - BLOCK_SIZE) pos_x = 0;

                for(int t = 0; t < nt; t++){
                    int blockmin_x = max(x_start, xx - t * neg_x);
                    int blockmin_y = max(y_start, yy - t * neg_y);
                    int blockmin_z = max(z_start, zz - t * neg_z);

                    int blockmax_x = max(x_start, xx + BLOCK_SIZE + t * pos_x);
                    int blockmax_y = max(y_start, yy + BLOCK_SIZE + t * pos_y);
                    int blockmax_z = max(z_start, zz + BLOCK_SIZE + t * pos_z);

                    dim3 grid_size (ceiling(blockmax_x - blockmin_x, 8),  
                                    ceiling(blockmax_y - blockmin_y, 8), 
                                    ceiling(blockmax_z - blockmin_z, 8));
                    dim3 block_size(8, 8, 8);
                    stencil_27_naive_kernel_1step<<<grid_size, block_size>>>(\
                        buffer[t % 2], buffer[(t + 1) % 2], blockmax_x - blockmin_x,\
                            blockmax_y - blockmin_y, blockmax_z - blockmin_z, \
                            blockmin_x, blockmin_y,\
                            blockmin_z, ldx, ldy);
                }
            }
        }
    }
    return buffer[nt % 2];
}