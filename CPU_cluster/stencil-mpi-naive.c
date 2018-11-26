#include "common.h"
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
#include <mpi.h>
const char* version_name = "mpi version";

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    /* multi-threads in many machines */
    grid_info->local_size_x = grid_info->global_size_x;
    grid_info->local_size_y = grid_info->global_size_y;
    grid_info->local_size_z = grid_info->global_size_z / grid_info->p_num;
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = grid_info->p_id * grid_info->local_size_z;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
    printf("thread id %d, %d %d\n", grid_info->p_id, grid_info->local_size_z, grid_info->offset_z);
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

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;


    int nthreads = 0;
    #pragma omp parallel
    {
        if(omp_get_thread_num() == 0) nthreads = omp_get_num_threads();
    }
    printf("We have %d threads\n", nthreads);

    for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];
        MPI_Status status;
        if(grid_info->p_id != grid_info->p_num - 1){
            MPI_Sendrecv((void *)&a0[INDEX(0, 0, z_end - 1, ldx, ldy)], ldx * ldy, MPI_DOUBLE, grid_info->p_id + 1, 0, \
                        (void *)&a0[INDEX(0, 0, z_end, ldx, ldy)], ldx * ldy, MPI_DOUBLE, grid_info->p_id + 1, 1, MPI_COMM_WORLD, &status);
        }
        if(grid_info->p_id != 0){
            MPI_Sendrecv((void *)&a0[INDEX(0, 0, z_start, ldx, ldy)], ldx * ldy, MPI_DOUBLE, grid_info->p_id - 1, 1, \
                        (void *)&a0[INDEX(0, 0, z_start - 1, ldx, ldy)], ldx * ldy, MPI_DOUBLE, grid_info->p_id - 1, 0, MPI_COMM_WORLD, &status);
        }
        int bs_y=16;
        if(y_end - y_start == 384) bs_y = 12;
        if(y_end - y_start == 512) bs_y = 8;
        for(int yy = y_start; yy < y_end; yy += bs_y){
            omp_set_num_threads(24);
            #pragma omp parallel for
            for(int z = z_start; z < z_end; ++z) {
                for(int y = yy; y < yy + bs_y; ++y) {
                    for(int x = x_start; x < x_end; x += 4) {
                        do_cal7(a0, a1, x, y, z, ldx, ldy);
                        // a1[INDEX(x, y, z, ldx, ldy)] \
                        //     = ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] \
                        //     + ALPHA_NZZ * a0[INDEX(x-1, y, z, ldx, ldy)] \
                        //     + ALPHA_PZZ * a0[INDEX(x+1, y, z, ldx, ldy)] \
                        //     + ALPHA_ZNZ * a0[INDEX(x, y-1, z, ldx, ldy)] \
                        //     + ALPHA_ZPZ * a0[INDEX(x, y+1, z, ldx, ldy)] \
                        //     + ALPHA_ZZN * a0[INDEX(x, y, z-1, ldx, ldy)] \
                        //     + ALPHA_ZZP * a0[INDEX(x, y, z+1, ldx, ldy)];
                    }
                }
            }        
        }
    }
    return buffer[nt % 2];
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

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    int nthreads = 0;
    #pragma omp parallel
    {
        if(omp_get_thread_num() == 0) nthreads = omp_get_num_threads();
    }
    printf("We have %d threads\n", nthreads);

    for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];
        MPI_Status status;
        if(grid_info->p_id != grid_info->p_num - 1){
            MPI_Sendrecv((void *)&a0[INDEX(0, 0, z_end - 1, ldx, ldy)], ldx * ldy, MPI_DOUBLE, grid_info->p_id + 1, 0, \
                        (void *)&a0[INDEX(0, 0, z_end, ldx, ldy)], ldx * ldy, MPI_DOUBLE, grid_info->p_id + 1, 1, MPI_COMM_WORLD, &status);
        }
        if(grid_info->p_id != 0){
            MPI_Sendrecv((void *)&a0[INDEX(0, 0, z_start, ldx, ldy)], ldx * ldy, MPI_DOUBLE, grid_info->p_id - 1, 1, \
                        (void *)&a0[INDEX(0, 0, z_start - 1, ldx, ldy)], ldx * ldy, MPI_DOUBLE, grid_info->p_id - 1, 0, MPI_COMM_WORLD, &status);
        }
        int bs_y=16;
        if(y_end - y_start == 384) bs_y = 12;
        if(y_end - y_start == 512) bs_y = 8;
        for(int yy = y_start; yy < y_end; yy += bs_y){
            omp_set_num_threads(24);
            #pragma omp parallel for
            for(int z = z_start; z < z_end; ++z) {
                for(int y = yy; y < yy + bs_y; ++y) {
                    for(int x = x_start; x < x_end; x += 4) {
                        do_cal27(a0, a1, x, y, z, ldx, ldy);
                        // a1[INDEX(x, y, z, ldx, ldy)] \
                        //     = ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] \
                        //     + ALPHA_NZZ * a0[INDEX(x-1, y, z, ldx, ldy)] \
                        //     + ALPHA_PZZ * a0[INDEX(x+1, y, z, ldx, ldy)] \
                        //     + ALPHA_ZNZ * a0[INDEX(x, y-1, z, ldx, ldy)] \
                        //     + ALPHA_ZPZ * a0[INDEX(x, y+1, z, ldx, ldy)] \
                        //     + ALPHA_ZZN * a0[INDEX(x, y, z-1, ldx, ldy)] \
                        //     + ALPHA_ZZP * a0[INDEX(x, y, z+1, ldx, ldy)];
                    }
                }
            }        
        }
    }
    return buffer[nt % 2];
}