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

__global__ void stencil_7_naive_kernel_1step(double * readQueuePlane0, double* readQueuePlane1, \
    double *readQueuePlane2, double *writeQueuePlane, \
    int nx, int ny, int halo_x, int halo_y, int z, \
    int writeOffset, int readOffset, int ldx, int ldy) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    if(tx < nx && ty < ny) {
        // int x = tx + halo_x;
        // int y = ty + halo_y;
        // writeQueuePlane[INDEX(x, y, z, ldx, ldy) - writeOffset] = 
        //     ALPHA_ZZZ * readQueuePlane1[INDEX(x, y, z, ldx, ldy) - readOffset] +
        //     ALPHA_ZZN * readQueuePlane0[INDEX(x, y, z, ldx, ldy) - readOffset] +
        //     ALPHA_ZZP * readQueuePlane2[INDEX(x, y, z, ldx, ldy) - readOffset] +
        //     ALPHA_ZNZ * readQueuePlane1[INDEX(x, y - 1, z, ldx, ldy) - readOffset] +
        //     ALPHA_NZZ * readQueuePlane1[INDEX(x - 1, y, z, ldx, ldy) - readOffset] +
        //     ALPHA_PZZ * readQueuePlane1[INDEX(x + 1, y, z, ldx, ldy) - readOffset] +
        //     ALPHA_ZPZ * readQueuePlane1[INDEX(x, y + 1, z, ldx, ldy) - readOffset];
    }
}

__global__ void stencil_27_naive_kernel_1step(double * readQueuePlane0, double* readQueuePlane1, \
    double *readQueuePlane2, double *writeQueuePlane, \
    int nx, int ny, int halo_x, int halo_y, int z, \
    int writeOffset, int readOffset, int ldx, int ldy) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    if(tx < nx && ty < ny) {
        int x = tx + halo_x;
        int y = ty + halo_y;
        writeQueuePlane[INDEX(x, y, z, ldx, ldy) - writeOffset] = 
            ALPHA_ZZZ * readQueuePlane1[INDEX(x, y, z, ldx, ldy) - readOffset] +
            ALPHA_ZZN * readQueuePlane0[INDEX(x, y, z, ldx, ldy) - readOffset] +
            ALPHA_ZZP * readQueuePlane2[INDEX(x, y, z, ldx, ldy) - readOffset] +
            ALPHA_ZNZ * readQueuePlane1[INDEX(x, y - 1, z, ldx, ldy) - readOffset] +
            ALPHA_NZZ * readQueuePlane1[INDEX(x - 1, y, z, ldx, ldy) - readOffset] +
            ALPHA_PZZ * readQueuePlane1[INDEX(x + 1, y, z, ldx, ldy) - readOffset] +
            ALPHA_ZPZ * readQueuePlane1[INDEX(x, y + 1, z, ldx, ldy) - readOffset];
    }
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    // int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    // int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    // int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    int ty = 32;
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
                    // dim3 grid_size (ceiling(ldx - 2, 8),  
                    //                 ceiling(writeBlockRealMax_y - writeBlockRealMin_y, 8));
                    // dim3 block_size(8, 8);
                    // stencil_7_naive_kernel_1step<<<grid_size, block_size>>>(\
                    //     readQueuePlane0, readQueuePlane1, readQueuePlane2, writeQueuePlane, \
                    //     ldx - 2, writeBlockRealMax_y - writeBlockRealMin_y, 1, writeBlockRealMin_y, k - t, \
                    //     writeOffset, readOffset, ldx, ldy);
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

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt) {
    ptr_t buffer[2] = {grid, aux};
    // int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    // int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    // int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

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
                    dim3 grid_size (ceiling(ldx - 2, 8),  
                                    ceiling(writeBlockRealMax_y - writeBlockRealMin_y, 8));
                    dim3 block_size(8, 8);
                    stencil_27_naive_kernel_1step<<<grid_size, block_size>>>(\
                        readQueuePlane0, readQueuePlane1, readQueuePlane2, writeQueuePlane, \
                        ldx - 2, writeBlockRealMax_y - writeBlockRealMin_y, 1, writeBlockRealMin_y, k - t, \
                        writeOffset, readOffset, ldx, ldy);
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