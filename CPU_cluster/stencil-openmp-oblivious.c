#include "common.h"
#include <omp.h>
#include <immintrin.h>

const char* version_name = "openmp version";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    /* multi-threads in one machine */
    if(grid_info->p_id == 0) {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    } else {
        grid_info->local_size_x = 0;
        grid_info->local_size_y = 0;
        grid_info->local_size_z = 0;
    }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {
    
}

void do_cal7(cptr_t a0, ptr_t a1, int x, int y, int z, int ldx, int ldy){
    __m256d res = _mm256_setzero_pd();
    __m256d in1 = _mm256_loadu_pd(a0 + INDEX(x, y, z, ldx, ldy));
    __m256d in2 = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z, ldx, ldy));
    __m256d in3 = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z, ldx, ldy));
    __m256d in4 = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z, ldx, ldy));
    __m256d in5 = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z, ldx, ldy));
    __m256d in6 = _mm256_loadu_pd(a0 + INDEX(x, y, z - 1, ldx, ldy));
    __m256d in7 = _mm256_loadu_pd(a0 + INDEX(x, y, z + 1, ldx, ldy));
    __m256d al1 = _mm256_set1_pd((double)ALPHA_ZZZ);
    __m256d al2 = _mm256_set1_pd((double)ALPHA_NZZ);
    __m256d al3 = _mm256_set1_pd((double)ALPHA_PZZ);
    __m256d al4 = _mm256_set1_pd((double)ALPHA_ZNZ);
    __m256d al5 = _mm256_set1_pd((double)ALPHA_ZPZ);
    __m256d al6 = _mm256_set1_pd((double)ALPHA_ZZN);
    __m256d al7 = _mm256_set1_pd((double)ALPHA_ZZP);
    res = _mm256_fmadd_pd(al1, in1, res);
    res = _mm256_fmadd_pd(al2, in2, res);
    res = _mm256_fmadd_pd(al3, in3, res);
    res = _mm256_fmadd_pd(al4, in4, res);
    res = _mm256_fmadd_pd(al5, in5, res);
    res = _mm256_fmadd_pd(al6, in6, res);
    res = _mm256_fmadd_pd(al7, in7, res);
    _mm256_storeu_pd(a1 + INDEX(x, y, z, ldx, ldy), res);
}

void do_cal27(cptr_t a0, ptr_t a1, int x, int y, int z, int ldx, int ldy){
    __m256d res = _mm256_setzero_pd();
    __m256d in1, in2, in3, in4, in5, in6, in7;
    __m256d al1, al2, al3, al4, al5, al6, al7;
    in1 = _mm256_loadu_pd(a0 + INDEX(x, y, z, ldx, ldy));
    in2 = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z, ldx, ldy));
    in3 = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z, ldx, ldy));
    in4 = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z, ldx, ldy));
    in5 = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z, ldx, ldy));
    in6 = _mm256_loadu_pd(a0 + INDEX(x, y, z - 1, ldx, ldy));
    in7 = _mm256_loadu_pd(a0 + INDEX(x, y, z + 1, ldx, ldy));
    al1 = _mm256_set1_pd((double)ALPHA_ZZZ);
    al2 = _mm256_set1_pd((double)ALPHA_NZZ);
    al3 = _mm256_set1_pd((double)ALPHA_PZZ);
    al4 = _mm256_set1_pd((double)ALPHA_ZNZ);
    al5 = _mm256_set1_pd((double)ALPHA_ZPZ);
    al6 = _mm256_set1_pd((double)ALPHA_ZZN);
    al7 = _mm256_set1_pd((double)ALPHA_ZZP);
    res = _mm256_fmadd_pd(al1, in1, res);
    res = _mm256_fmadd_pd(al2, in2, res);
    res = _mm256_fmadd_pd(al3, in3, res);
    res = _mm256_fmadd_pd(al4, in4, res);
    res = _mm256_fmadd_pd(al5, in5, res);
    res = _mm256_fmadd_pd(al6, in6, res);
    res = _mm256_fmadd_pd(al7, in7, res);

    in1 = _mm256_loadu_pd(a0 + INDEX(x - 1, y - 1, z, ldx, ldy));
    in2 = _mm256_loadu_pd(a0 + INDEX(x + 1, y - 1, z, ldx, ldy));
    in3 = _mm256_loadu_pd(a0 + INDEX(x - 1, y + 1, z, ldx, ldy));
    in4 = _mm256_loadu_pd(a0 + INDEX(x + 1, y + 1, z, ldx, ldy));
    in5 = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z - 1, ldx, ldy));
    in6 = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z - 1, ldx, ldy));
    in7 = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z + 1, ldx, ldy));
    al1 = _mm256_set1_pd((double)ALPHA_NNZ);
    al2 = _mm256_set1_pd((double)ALPHA_PNZ);
    al3 = _mm256_set1_pd((double)ALPHA_NPZ);
    al4 = _mm256_set1_pd((double)ALPHA_PPZ);
    al5 = _mm256_set1_pd((double)ALPHA_NZN);
    al6 = _mm256_set1_pd((double)ALPHA_PZN);
    al7 = _mm256_set1_pd((double)ALPHA_NZP);
    res = _mm256_fmadd_pd(al1, in1, res);
    res = _mm256_fmadd_pd(al2, in2, res);
    res = _mm256_fmadd_pd(al3, in3, res);
    res = _mm256_fmadd_pd(al4, in4, res);
    res = _mm256_fmadd_pd(al5, in5, res);
    res = _mm256_fmadd_pd(al6, in6, res);
    res = _mm256_fmadd_pd(al7, in7, res);

    in1 = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z + 1, ldx, ldy));
    in2 = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z - 1, ldx, ldy));
    in3 = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z - 1, ldx, ldy));
    in4 = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z + 1, ldx, ldy));
    in5 = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z + 1, ldx, ldy));
    in6 = _mm256_loadu_pd(a0 + INDEX(x - 1, y - 1, z - 1, ldx, ldy));
    in7 = _mm256_loadu_pd(a0 + INDEX(x + 1, y - 1, z - 1, ldx, ldy));
    al1 = _mm256_set1_pd((double)ALPHA_PZP);
    al2 = _mm256_set1_pd((double)ALPHA_ZNN);
    al3 = _mm256_set1_pd((double)ALPHA_ZPN);
    al4 = _mm256_set1_pd((double)ALPHA_ZNP);
    al5 = _mm256_set1_pd((double)ALPHA_ZPP);
    al6 = _mm256_set1_pd((double)ALPHA_NNN);
    al7 = _mm256_set1_pd((double)ALPHA_PNN);
    res = _mm256_fmadd_pd(al1, in1, res);
    res = _mm256_fmadd_pd(al2, in2, res);
    res = _mm256_fmadd_pd(al3, in3, res);
    res = _mm256_fmadd_pd(al4, in4, res);
    res = _mm256_fmadd_pd(al5, in5, res);
    res = _mm256_fmadd_pd(al6, in6, res);
    res = _mm256_fmadd_pd(al7, in7, res);

    in1 = _mm256_loadu_pd(a0 + INDEX(x - 1, y + 1, z - 1, ldx, ldy));
    in2 = _mm256_loadu_pd(a0 + INDEX(x + 1, y + 1, z - 1, ldx, ldy));
    in3 = _mm256_loadu_pd(a0 + INDEX(x - 1, y - 1, z + 1, ldx, ldy));
    in4 = _mm256_loadu_pd(a0 + INDEX(x + 1, y - 1, z + 1, ldx, ldy));
    in5 = _mm256_loadu_pd(a0 + INDEX(x - 1, y + 1, z + 1, ldx, ldy));
    in6 = _mm256_loadu_pd(a0 + INDEX(x + 1, y + 1, z + 1, ldx, ldy));
    al1 = _mm256_set1_pd((double)ALPHA_NPN);
    al2 = _mm256_set1_pd((double)ALPHA_PPN);
    al3 = _mm256_set1_pd((double)ALPHA_NNP);
    al4 = _mm256_set1_pd((double)ALPHA_PNP);
    al5 = _mm256_set1_pd((double)ALPHA_NPP);
    al6 = _mm256_set1_pd((double)ALPHA_PPP);
    res = _mm256_fmadd_pd(al1, in1, res);
    res = _mm256_fmadd_pd(al2, in2, res);
    res = _mm256_fmadd_pd(al3, in3, res);
    res = _mm256_fmadd_pd(al4, in4, res);
    res = _mm256_fmadd_pd(al5, in5, res);
    res = _mm256_fmadd_pd(al6, in6, res);

    _mm256_storeu_pd(a1 + INDEX(x, y, z, ldx, ldy), res);
}

void walk3(ptr_t buffer[], int nx, int ny, int nz, int t0, int t1,
            int type, int x0, int dx0, int x1, int dx1,
                      int y0, int dy0, int y1, int dy1,
                      int z0, int dz0, int z1, int dz1){
    int dt = t1 - t0;
    if(dt == 1 || (x1-x0)*(y1-y0)*(z1-z0) < 2097152){
        for(int t = t0; t < t1; t++){
            cptr_t a0 = buffer[t % 2];
            ptr_t a1 = buffer[(t + 1) % 2];
            omp_set_num_threads(24);
            #pragma omp parallel for
            for(int z = z0 + dz0 * (t - t0); z < z1 + dz1 * (t - t0); z++){
                for(int y = y0 + dy0 * (t - t0); y < y1 + dy1 * (t - t0); y++){
                    for(int x = x0 + dx0 * (t - t0); x < x1 + dx1 * (t - t0); x += 4){
                        if(type == 7) do_cal7(a0, a1, x, y, z, nx, ny);
                        else do_cal27(a0, a1, x, y, z, nx, ny);
                    }
                    for(int x = (x1 + dx1 * (t - t0))/ 4 * 4; x < x1 + dx1 * (t - t0); x++){
                        if(type == 7){
                            a1[INDEX(x, y, z, nx, ny)] \
                            = ALPHA_ZZZ * a0[INDEX(x, y, z, nx, ny)] \
                            + ALPHA_NZZ * a0[INDEX(x-1, y, z, nx, ny)] \
                            + ALPHA_PZZ * a0[INDEX(x+1, y, z, nx, ny)] \
                            + ALPHA_ZNZ * a0[INDEX(x, y-1, z, nx, ny)] \
                            + ALPHA_ZPZ * a0[INDEX(x, y+1, z, nx, ny)] \
                            + ALPHA_ZZN * a0[INDEX(x, y, z-1, nx, ny)] \
                            + ALPHA_ZZP * a0[INDEX(x, y, z+1, nx, ny)];
                        }
                        else{
                            a1[INDEX(x, y, z, nx, ny)] \
                            = ALPHA_ZZZ * a0[INDEX(x, y, z, nx, ny)] \
                            + ALPHA_NZZ * a0[INDEX(x-1, y, z, nx, ny)] \
                            + ALPHA_PZZ * a0[INDEX(x+1, y, z, nx, ny)] \
                            + ALPHA_ZNZ * a0[INDEX(x, y-1, z, nx, ny)] \
                            + ALPHA_ZPZ * a0[INDEX(x, y+1, z, nx, ny)] \
                            + ALPHA_ZZN * a0[INDEX(x, y, z-1, nx, ny)] \
                            + ALPHA_ZZP * a0[INDEX(x, y, z+1, nx, ny)] \
                            + ALPHA_NNZ * a0[INDEX(x-1, y-1, z, nx, ny)] \
                            + ALPHA_PNZ * a0[INDEX(x+1, y-1, z, nx, ny)] \
                            + ALPHA_NPZ * a0[INDEX(x-1, y+1, z, nx, ny)] \
                            + ALPHA_PPZ * a0[INDEX(x+1, y+1, z, nx, ny)] \
                            + ALPHA_NZN * a0[INDEX(x-1, y, z-1, nx, ny)] \
                            + ALPHA_PZN * a0[INDEX(x+1, y, z-1, nx, ny)] \
                            + ALPHA_NZP * a0[INDEX(x-1, y, z+1, nx, ny)] \
                            + ALPHA_PZP * a0[INDEX(x+1, y, z+1, nx, ny)] \
                            + ALPHA_ZNN * a0[INDEX(x, y-1, z-1, nx, ny)] \
                            + ALPHA_ZPN * a0[INDEX(x, y+1, z-1, nx, ny)] \
                            + ALPHA_ZNP * a0[INDEX(x, y-1, z+1, nx, ny)] \
                            + ALPHA_ZPP * a0[INDEX(x, y+1, z+1, nx, ny)] \
                            + ALPHA_NNN * a0[INDEX(x-1, y-1, z-1, nx, ny)] \
                            + ALPHA_PNN * a0[INDEX(x+1, y-1, z-1, nx, ny)] \
                            + ALPHA_NPN * a0[INDEX(x-1, y+1, z-1, nx, ny)] \
                            + ALPHA_PPN * a0[INDEX(x+1, y+1, z-1, nx, ny)] \
                            + ALPHA_NNP * a0[INDEX(x-1, y-1, z+1, nx, ny)] \
                            + ALPHA_PNP * a0[INDEX(x+1, y-1, z+1, nx, ny)] \
                            + ALPHA_NPP * a0[INDEX(x-1, y+1, z+1, nx, ny)] \
                            + ALPHA_PPP * a0[INDEX(x+1, y+1, z+1, nx, ny)];
                        }
                    }
                }
            }
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