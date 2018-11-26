#include "common.h"
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
const char* version_name = "openmp version";

double *queuePlanes, *queuePlane0, *queuePlane1, *queuePlane2;
int *queuePlanesIndices;

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

void CircularQueueInit(int nx, int ty, int timesteps) {
    int numPointsInQueuePlane, t;
  
    queuePlanesIndices = (int *)malloc((timesteps-1) * sizeof(int));
  
    if (queuePlanesIndices==NULL) {
        printf("Error on array queuePlanesIndices malloc.\n");
        exit(EXIT_FAILURE);
    }
    
    int queuePlanesIndexPtr = 0;
    
    for (t=1; t < timesteps; t++) {
        queuePlanesIndices[t-1] = queuePlanesIndexPtr;
        numPointsInQueuePlane = (ty+2*(timesteps-t)) * nx;
        queuePlanesIndexPtr += numPointsInQueuePlane;
    }

    queuePlanes = (double *)malloc(3 * queuePlanesIndexPtr * sizeof(double));
    
    if (queuePlanes==NULL) {
        printf("Error on array queuePlanes malloc.\n");
        exit(EXIT_FAILURE);
    }
    
    queuePlane0 = queuePlanes;
    queuePlane1 = &queuePlanes[queuePlanesIndexPtr];
    queuePlane2 = &queuePlanes[2 * queuePlanesIndexPtr];
}

void do_cal7(ptr_t a0, ptr_t a1, ptr_t a2, ptr_t a3, int x, int y, int z, int ldx, int ldy, int woff, int roff){
    __m256d res = _mm256_setzero_pd();
    __m256d in1 = _mm256_loadu_pd(a1 + INDEX(x, y, z, ldx, ldy) - roff);
    __m256d in2 = _mm256_loadu_pd(a1 + INDEX(x - 1, y, z, ldx, ldy) - roff);
    __m256d in3 = _mm256_loadu_pd(a1 + INDEX(x + 1, y, z, ldx, ldy) - roff);
    __m256d in4 = _mm256_loadu_pd(a1 + INDEX(x, y - 1, z, ldx, ldy) - roff);
    __m256d in5 = _mm256_loadu_pd(a1 + INDEX(x, y + 1, z, ldx, ldy) - roff);
    __m256d in6 = _mm256_loadu_pd(a0 + INDEX(x, y, z, ldx, ldy) - roff);
    __m256d in7 = _mm256_loadu_pd(a2 + INDEX(x, y, z, ldx, ldy) - roff);
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
    _mm256_storeu_pd(a3 + INDEX(x, y, z, ldx, ldy) - woff, res);
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
    omp_set_num_threads(24);
    #pragma omp parallel
    {
        if(omp_get_thread_num() == 0) nthreads = omp_get_num_threads();
    }
    printf("We have %d threads\n", nthreads);

    int ty = ldy-2;
    int numBlocks_y = (ldy - 2) / ty;
    double * a0 = (double *)buffer[0];
    double * a1 = (double *)buffer[1];

    CircularQueueInit(ldx, ty, nt);
    int t;
    for (int s = 0; s < numBlocks_y; s++) {
        for (int k = 1; k < (ldz + nt - 2); k++) {
            double *tempQueuePlane;
            //#pragma omp parallel for
            for (t = 0; t < nt; t++) {
                double *readQueuePlane0, *readQueuePlane1, *readQueuePlane2, *writeQueuePlane;
                if ((k > t) && (k < (ldz - 1 + t))) {    
                    if (t == 0) {
                        readQueuePlane0 = &a0[INDEX(0, 0, k-1, ldx, ldy)];
                        readQueuePlane1 = &a0[INDEX(0, 0, k, ldx, ldy)];
                        readQueuePlane2 = &a0[INDEX(0, 0, k+1, ldx, ldy)];
                    }
                    else {
                        readQueuePlane0 = &queuePlane0[queuePlanesIndices[t - 1]];
                        readQueuePlane1 = &queuePlane1[queuePlanesIndices[t - 1]];
                        readQueuePlane2 = &queuePlane2[queuePlanesIndices[t - 1]];
                    }

                    // determine the edges of the queues
                    int writeBlockMin_y = s * ty - (nt - t) + 2;
                    int writeBlockMax_y = (s + 1) * ty + (nt - t);
                    int writeBlockRealMin_y = writeBlockMin_y;
                    int writeBlockRealMax_y = writeBlockMax_y;

                    if (writeBlockMin_y < 1) {
                        writeBlockMin_y = 0;
                        writeBlockRealMin_y = 1;
                    }
                    if (writeBlockMax_y > (ldy - 1)) {
                        writeBlockMax_y = ldy;
                        writeBlockRealMax_y = ldy-1;
                    }

                    int writeOffset;
                    if (t == (nt - 1)) {
                        writeQueuePlane = a1;
                        writeOffset = 0;
                    }
                    else {
                        writeQueuePlane = &queuePlane2[queuePlanesIndices[t]];
                        writeOffset = INDEX(0, writeBlockMin_y, k - t, ldx, ldy);
                    }

                    int readOffset;
                    if ((writeBlockMin_y == 0) || (t == 0)) {
                        readOffset = INDEX(0, 0, k - t, ldx, ldy);
                    }
                    else {
                        readOffset = INDEX(0, writeBlockMin_y - 1, k - t, ldx, ldy);
                    }

                    // use ghost cells for the bottommost and topmost planes
                    if (k == (t+1)) {
                        readQueuePlane0 = a0;
                    }
                    if (k == (ldz + t - 2)) {
                        readQueuePlane2 = &a0[INDEX(0, 0, ldz - 1, ldx, ldy)];
                    }

                    // copy ghost cells
                    if (t < (nt - 1)) {
                        for (int j = (writeBlockMin_y + 1); j < (writeBlockMax_y - 1); j++) {
                            writeQueuePlane[INDEX(0, j, k - t, ldx, ldy) - writeOffset] = readQueuePlane1[INDEX(0, j, k - t, ldx, ldy) - readOffset];
                            writeQueuePlane[INDEX(ldx - 1, j, k - t, ldx, ldy) - writeOffset] = readQueuePlane1[INDEX(ldx - 1, j, k - t, ldx, ldy) - readOffset];
                        }
                        if (writeBlockMin_y == 0) {
                            memcpy(writeQueuePlane + INDEX(1, writeBlockMin_y, k - t, ldx, ldy) - writeOffset, \
                            readQueuePlane1 + INDEX(1, writeBlockMin_y, k - t, ldx, ldy) - readOffset, (ldx - 2) * sizeof(double));
                        }
                        if (writeBlockMax_y == ldy) {
                            memcpy(writeQueuePlane + INDEX(1, writeBlockRealMax_y, k - t, ldx, ldy) - writeOffset, \
                            readQueuePlane1 + INDEX(1, writeBlockRealMax_y, k - t, ldx, ldy) - readOffset, (ldx - 2) * sizeof(double));
                        }
                    }

                    // actual calculations
                    #pragma omp parallel for
                    for (int j = writeBlockRealMin_y; j < writeBlockRealMax_y; j++) {
                        for (int i = 1; i < 1 + (ldx - 2) / 4 * 4; i += 4) {
                            do_cal7(readQueuePlane0, readQueuePlane1, readQueuePlane2, writeQueuePlane, i, j, k - t, ldx, ldy, writeOffset, readOffset);
                            // writeQueuePlane[INDEX(i, j, k - t, ldx, ldy) - writeOffset] = 
                            // ALPHA_ZZZ * readQueuePlane1[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_ZZN * readQueuePlane0[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_ZZP * readQueuePlane2[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_ZNZ * readQueuePlane1[INDEX(i, j - 1, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_NZZ * readQueuePlane1[INDEX(i - 1, j, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_PZZ * readQueuePlane1[INDEX(i + 1, j, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_ZPZ * readQueuePlane1[INDEX(i, j + 1, k - t, ldx, ldy) - readOffset];
                        }
                        for(int i = 1 + (ldx - 2) / 4 * 4; i < (ldx - 1); i++){
                            writeQueuePlane[INDEX(i, j, k - t, ldx, ldy) - writeOffset] = 
                            ALPHA_ZZZ * readQueuePlane1[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                            ALPHA_ZZN * readQueuePlane0[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                            ALPHA_ZZP * readQueuePlane2[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                            ALPHA_ZNZ * readQueuePlane1[INDEX(i, j - 1, k - t, ldx, ldy) - readOffset] +
                            ALPHA_NZZ * readQueuePlane1[INDEX(i - 1, j, k - t, ldx, ldy) - readOffset] +
                            ALPHA_PZZ * readQueuePlane1[INDEX(i + 1, j, k - t, ldx, ldy) - readOffset] +
                            ALPHA_ZPZ * readQueuePlane1[INDEX(i, j + 1, k - t, ldx, ldy) - readOffset];
                        }
                    }
                }
            }
            if(t > 0){
                tempQueuePlane = queuePlane0;
                queuePlane0 = queuePlane1;
                queuePlane1 = queuePlane2;
                queuePlane2 = tempQueuePlane;   
            }
        }
    }
    free(queuePlanesIndices);
    free(queuePlanes);
    return buffer[1];
}

void do_cal27(ptr_t a0, ptr_t a1, ptr_t a2, ptr_t a3, int x, int y, int z, int ldx, int ldy, int woff, int roff){
    __m256d res = _mm256_setzero_pd();
    __m256d in1, in2, in3, in4, in5, in6, in7;
    __m256d al1, al2, al3, al4, al5, al6, al7;
    in1 = _mm256_loadu_pd(a1 + INDEX(x, y, z, ldx, ldy) - roff);
    in2 = _mm256_loadu_pd(a1 + INDEX(x - 1, y, z, ldx, ldy) - roff);
    in3 = _mm256_loadu_pd(a1 + INDEX(x + 1, y, z, ldx, ldy) - roff);
    in4 = _mm256_loadu_pd(a1 + INDEX(x, y - 1, z, ldx, ldy) - roff);
    in5 = _mm256_loadu_pd(a1 + INDEX(x, y + 1, z, ldx, ldy) - roff);
    in6 = _mm256_loadu_pd(a0 + INDEX(x, y, z, ldx, ldy) - roff);
    in7 = _mm256_loadu_pd(a2 + INDEX(x, y, z, ldx, ldy) - roff);
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

    in1 = _mm256_loadu_pd(a1 + INDEX(x - 1, y - 1, z, ldx, ldy) - roff);
    in2 = _mm256_loadu_pd(a1 + INDEX(x + 1, y - 1, z, ldx, ldy) - roff);
    in3 = _mm256_loadu_pd(a1 + INDEX(x - 1, y + 1, z, ldx, ldy) - roff);
    in4 = _mm256_loadu_pd(a1 + INDEX(x + 1, y + 1, z, ldx, ldy) - roff);
    in5 = _mm256_loadu_pd(a0 + INDEX(x - 1, y, z, ldx, ldy) - roff);
    in6 = _mm256_loadu_pd(a0 + INDEX(x + 1, y, z, ldx, ldy) - roff);
    in7 = _mm256_loadu_pd(a2 + INDEX(x - 1, y, z, ldx, ldy) - roff);
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

    in1 = _mm256_loadu_pd(a2 + INDEX(x + 1, y, z, ldx, ldy) - roff);
    in2 = _mm256_loadu_pd(a0 + INDEX(x, y - 1, z, ldx, ldy) - roff);
    in3 = _mm256_loadu_pd(a0 + INDEX(x, y + 1, z, ldx, ldy) - roff);
    in4 = _mm256_loadu_pd(a2 + INDEX(x, y - 1, z, ldx, ldy) - roff);
    in5 = _mm256_loadu_pd(a2 + INDEX(x, y + 1, z, ldx, ldy) - roff);
    in6 = _mm256_loadu_pd(a0 + INDEX(x - 1, y - 1, z, ldx, ldy) - roff);
    in7 = _mm256_loadu_pd(a0 + INDEX(x + 1, y - 1, z, ldx, ldy) - roff);
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

    in1 = _mm256_loadu_pd(a0 + INDEX(x - 1, y + 1, z, ldx, ldy) - roff);
    in2 = _mm256_loadu_pd(a0 + INDEX(x + 1, y + 1, z, ldx, ldy) - roff);
    in3 = _mm256_loadu_pd(a2 + INDEX(x - 1, y - 1, z, ldx, ldy) - roff);
    in4 = _mm256_loadu_pd(a2 + INDEX(x + 1, y - 1, z, ldx, ldy) - roff);
    in5 = _mm256_loadu_pd(a2 + INDEX(x - 1, y + 1, z, ldx, ldy) - roff);
    in6 = _mm256_loadu_pd(a2 + INDEX(x + 1, y + 1, z, ldx, ldy) - roff);
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

    _mm256_storeu_pd(a3 + INDEX(x, y, z, ldx, ldy) - woff, res);
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
    omp_set_num_threads(24);
    #pragma omp parallel
    {
        if(omp_get_thread_num() == 0) nthreads = omp_get_num_threads();
    }
    printf("We have %d threads\n", nthreads);

    int ty = ldy - 2;
    int numBlocks_y = (ldy - 2) / ty;
    double * a0 = (double *)buffer[0];
    double * a1 = (double *)buffer[1];

    CircularQueueInit(ldx, ty, nt);
    int t;
    for (int s = 0; s < numBlocks_y; s++) {
        for (int k = 1; k < (ldz + nt - 2); k++) {
            double *tempQueuePlane;
            for (t = 0; t < nt; t++) {
                double *readQueuePlane0, *readQueuePlane1, *readQueuePlane2, *writeQueuePlane;
                if ((k > t) && (k < (ldz - 1 + t))) {    
                    if (t == 0) {
                        readQueuePlane0 = &a0[INDEX(0, 0, k-1, ldx, ldy)];
                        readQueuePlane1 = &a0[INDEX(0, 0, k, ldx, ldy)];
                        readQueuePlane2 = &a0[INDEX(0, 0, k+1, ldx, ldy)];
                    }
                    else {
                        readQueuePlane0 = &queuePlane0[queuePlanesIndices[t - 1]];
                        readQueuePlane1 = &queuePlane1[queuePlanesIndices[t - 1]];
                        readQueuePlane2 = &queuePlane2[queuePlanesIndices[t - 1]];
                    }

                    // determine the edges of the queues
                    int writeBlockMin_y = s * ty - (nt - t) + 2;
                    int writeBlockMax_y = (s + 1) * ty + (nt - t);
                    int writeBlockRealMin_y = writeBlockMin_y;
                    int writeBlockRealMax_y = writeBlockMax_y;

                    if (writeBlockMin_y < 1) {
                        writeBlockMin_y = 0;
                        writeBlockRealMin_y = 1;
                    }
                    if (writeBlockMax_y > (ldy - 1)) {
                        writeBlockMax_y = ldy;
                        writeBlockRealMax_y = ldy-1;
                    }

                    int writeOffset;
                    if (t == (nt - 1)) {
                        writeQueuePlane = a1;
                        writeOffset = 0;
                    }
                    else {
                        writeQueuePlane = &queuePlane2[queuePlanesIndices[t]];
                        writeOffset = INDEX(0, writeBlockMin_y, k - t, ldx, ldy);
                    }

                    int readOffset;
                    if ((writeBlockMin_y == 0) || (t == 0)) {
                        readOffset = INDEX(0, 0, k - t, ldx, ldy);
                    }
                    else {
                        readOffset = INDEX(0, writeBlockMin_y - 1, k - t, ldx, ldy);
                    }

                    // use ghost cells for the bottommost and topmost planes
                    if (k == (t+1)) {
                        readQueuePlane0 = a0;
                    }
                    if (k == (ldz + t - 2)) {
                        readQueuePlane2 = &a0[INDEX(0, 0, ldz-1, ldx, ldy)];
                    }

                    // copy ghost cells
                    if (t < (nt - 1)) {
                        for (int j = (writeBlockMin_y + 1); j < (writeBlockMax_y - 1); j++) {
                            writeQueuePlane[INDEX(0, j, k-t, ldx, ldy) - writeOffset] = readQueuePlane1[INDEX(0, j, k - t, ldx, ldy) - readOffset];
                            writeQueuePlane[INDEX(ldx - 1, j, k - t, ldx, ldy) - writeOffset] = readQueuePlane1[INDEX(ldx - 1, j, k - t, ldx, ldy) - readOffset];
                        }
                        if (writeBlockMin_y == 0) {
                            memcpy(writeQueuePlane + INDEX(1, writeBlockMin_y, k - t, ldx, ldy) - writeOffset, \
                            readQueuePlane1 + INDEX(1, writeBlockMin_y, k - t, ldx, ldy) - readOffset, (ldx - 2) * sizeof(double));
                        }
                        if (writeBlockMax_y == ldy) {
                            memcpy(writeQueuePlane + INDEX(1, writeBlockRealMax_y, k - t, ldx, ldy) - writeOffset, \
                            readQueuePlane1 + INDEX(1, writeBlockRealMax_y, k - t, ldx, ldy) - readOffset, (ldx - 2) * sizeof(double));
                        }
                    }

                    // actual calculations
                    #pragma omp parallel for
                    for (int j = writeBlockRealMin_y; j < writeBlockRealMax_y; j++) {
                        for (int i = 1; i < (ldx - 1); i += 4) {
                            do_cal27(readQueuePlane0, readQueuePlane1, readQueuePlane2, writeQueuePlane, i, j, k - t, ldx, ldy, writeOffset, readOffset);
                            // writeQueuePlane[INDEX(i, j, k - t, ldx, ldy) - writeOffset] = 
                            // ALPHA_ZZZ * readQueuePlane1[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_ZZN * readQueuePlane0[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_ZZP * readQueuePlane2[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_ZNZ * readQueuePlane1[INDEX(i, j - 1, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_NZZ * readQueuePlane1[INDEX(i - 1, j, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_PZZ * readQueuePlane1[INDEX(i + 1, j, k - t, ldx, ldy) - readOffset] +
                            // ALPHA_ZPZ * readQueuePlane1[INDEX(i, j + 1, k - t, ldx, ldy) - readOffset];
                        }
                        // for(int i = 1 + (ldx - 2) / 4 * 4; i < (ldx - 1); i++){
                        //     writeQueuePlane[INDEX(i, j, k - t, ldx, ldy) - writeOffset] = 
                        //     ALPHA_ZZZ * readQueuePlane1[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                        //     ALPHA_ZZN * readQueuePlane0[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                        //     ALPHA_ZZP * readQueuePlane2[INDEX(i, j, k - t, ldx, ldy) - readOffset] +
                        //     ALPHA_ZNZ * readQueuePlane1[INDEX(i, j - 1, k - t, ldx, ldy) - readOffset] +
                        //     ALPHA_NZZ * readQueuePlane1[INDEX(i - 1, j, k - t, ldx, ldy) - readOffset] +
                        //     ALPHA_PZZ * readQueuePlane1[INDEX(i + 1, j, k - t, ldx, ldy) - readOffset] +
                        //     ALPHA_ZPZ * readQueuePlane1[INDEX(i, j + 1, k - t, ldx, ldy) - readOffset];
                        // }
                    }
                }
            }
            if(t > 0){
                tempQueuePlane = queuePlane0;
                queuePlane0 = queuePlane1;
                queuePlane1 = queuePlane2;
                queuePlane2 = tempQueuePlane;  
            } 
        }
    }
    free(queuePlanesIndices);
    free(queuePlanes);
    return buffer[1];
}