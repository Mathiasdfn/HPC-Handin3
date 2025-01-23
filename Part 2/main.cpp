/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include "alloc3d.h"
#include "alloc3d_dev.h"
#include "jacobi.h"
#include "jacobi_offload.h"
#include "jacobi_offload_map.h"
#include "jacobi_offload_dual.h"

void init_f(double ***f, int N);
void init_u_inner(double ***u, int N, double start_T);
void init_u_borders(double ***u, int N);
void init_u_corners(double ***u, int N);
double diff_norm_squared(double ***u1, double ***u2, int N);

int main(int argc, char *argv[]) {
    // get the paramters from the command line
    int 	N = 500;
    int 	iter_max = 1000;
    double	tolerance = -1.0;
    double	start_T = 10;
    bool    debug_print = false;
    bool    host = true;
    bool    offload_map = true;
    bool    offload = true;
    bool    offload_dual = true;

    if (argc >= 2) N = atoi(argv[1]);	// grid size
    if (argc >= 3) iter_max = atoi(argv[2]);  // max. no. of iterations
    if (argc >= 4) tolerance = atof(argv[3]);  // tolerance (negative to disable)
    if (argc >= 5) start_T = atof(argv[4]);  // start T for all inner grid points
    if (argc >= 6) debug_print = atof(argv[5]);  // do debug print statements
    if (argc >= 7) offload_map = atof(argv[6]);  // do offload map
    if (argc >= 8) offload = atof(argv[7]);  // do offload
    if (argc >= 9) offload_dual = atof(argv[8]);  // do debug print dual

    // needed variables
    double tol = tolerance;
    int iter;
    double  start_time, end_time, time;

    // allocate memory
    if(debug_print) printf("Allocate memory on host\n");
    double  ***f = NULL;
    double 	***u = NULL;
    double 	***u_old = NULL;
    double 	***u2 = NULL;
    double 	***u_old2 = NULL;

    if ( (f = malloc_3d(N, N, N)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }
    if ( (u = malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (u_old = malloc_3d(N, N, N)) == NULL ) {
        perror("array u_old: allocation failed");
        exit(-1);
    }
    if ( (u2 = malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (u_old2 = malloc_3d(N, N, N)) == NULL ) {
        perror("array u_old: allocation failed");
        exit(-1);
    }


    // allocate memory on device
    double  ***f_d = NULL;
    double 	***u_d = NULL;
    double 	***u_old_d = NULL;
    double  *data_f_d = NULL;
    double 	*data_u_d = NULL;
    double 	*data_u_old_d = NULL;

    double  ***f_d0 = NULL;
    double 	***u_d0 = NULL;
    double 	***u_old_d0 = NULL;
    double  *data_f_d0 = NULL;
    double 	*data_u_d0 = NULL;
    double 	*data_u_old_d0 = NULL;

    double  ***f_d1 = NULL;
    double 	***u_d1 = NULL;
    double 	***u_old_d1 = NULL;
    double  *data_f_d1 = NULL;
    double 	*data_u_d1 = NULL;
    double 	*data_u_old_d1 = NULL;

    if(offload) {
        if(debug_print) printf("Allocate memory on device\n");

        if ( (f_d = d_malloc_3d(N, N, N, &data_f_d)) == NULL ) {
            perror("array f: allocation failed");
            exit(-1);
        }
        if ( (u_d = d_malloc_3d(N, N, N, &data_u_d)) == NULL ) {
            perror("array u: allocation failed");
            exit(-1);
        }
        if ( (u_old_d = d_malloc_3d(N, N, N, &data_u_old_d)) == NULL ) {
            perror("array u_old: allocation failed");
            exit(-1);
        }
    }
    if(offload_dual) {
        if(debug_print) printf("Allocate memory on two devices\n");
        
        omp_set_default_device(0);
        if ( (f_d0 = d_malloc_3d(N/2, N, N, &data_f_d0)) == NULL ) {
            perror("array f: allocation failed");
            exit(-1);
        }
        if ( (u_d0 = d_malloc_3d(N/2, N, N, &data_u_d0)) == NULL ) {
            perror("array u: allocation failed");
            exit(-1);
        }
        if ( (u_old_d0 = d_malloc_3d(N/2, N, N, &data_u_old_d0)) == NULL ) {
            perror("array u_old: allocation failed");
            exit(-1);
        }

        omp_set_default_device(1);
        if ( (f_d1 = d_malloc_3d(N/2, N, N, &data_f_d1)) == NULL ) {
            perror("array f: allocation failed");
            exit(-1);
        }
        if ( (u_d1 = d_malloc_3d(N/2, N, N, &data_u_d1)) == NULL ) {
            perror("array u: allocation failed");
            exit(-1);
        }
        if ( (u_old_d1 = d_malloc_3d(N/2, N, N, &data_u_old_d1)) == NULL ) {
            perror("array u_old: allocation failed");
            exit(-1);
        }

        omp_set_default_device(0);
    }


    // initialize matrices on host
    if(debug_print) printf("Initialize matrices on host\n");
    init_f(f, N);
    init_u_inner(u, N, start_T);
    init_u_borders(u, N);
    init_u_borders(u_old, N);
    init_u_inner(u2, N, start_T);
    init_u_borders(u2, N);
    init_u_borders(u_old2, N);


    // device number and host number
    int dev_num = omp_get_default_device();
    int host_num = omp_get_initial_device();
    if(debug_print) printf("Host number: %d and device number: %d\n", host_num, dev_num);


    // copy memory to device
    if(offload) {
        if(debug_print) printf("Copy memory to device\n");

        omp_target_memcpy(data_f_d, f[0][0], N * N * N * sizeof(double), 0, 0, dev_num, host_num);
        omp_target_memcpy(data_u_d, u[0][0], N * N * N * sizeof(double), 0, 0, dev_num, host_num);
        omp_target_memcpy(data_u_old_d, u_old[0][0], N * N * N * sizeof(double), 0, 0, dev_num, host_num);
    }


    // copy memory to two devices   
    if(offload_dual) {
        if(debug_print) printf("Copy memory to two devices\n");

        omp_target_memcpy(data_f_d0, f[0][0], N * N * N * sizeof(double) / 2, 0, 0, 0, host_num);
        omp_target_memcpy(data_u_d0, u[0][0], N * N * N * sizeof(double) / 2, 0, 0, 0, host_num);
        omp_target_memcpy(data_u_old_d0, u_old[0][0], N * N * N * sizeof(double) / 2, 0, 0, 0, host_num);

        omp_target_memcpy(data_f_d1, f[N/2][0], N * N * N * sizeof(double) / 2, 0, 0, 1, host_num);
        omp_target_memcpy(data_u_d1, u[N/2][0], N * N * N * sizeof(double) / 2, 0, 0, 1, host_num);
        omp_target_memcpy(data_u_old_d1, u_old[N/2][0], N * N * N * sizeof(double) / 2, 0, 0, 1, host_num);
    }


    // compute and time jacobi on host
    if(host) {
        if(debug_print) printf("Compute jacobi on host\n");

        start_time = omp_get_wtime();
        if (tolerance >= 0) {
            iter = jacobi_tol(f, u, u_old, N, iter_max, &tol);
        } else {
            jacobi(f, u, u_old, N, iter_max);
        }
        end_time = omp_get_wtime();

        printf("Time for jacobi: %.6f seconds\n", end_time - start_time);
        if (tolerance >= 0) {
            printf("Iter for jacobi: %d\n", iter);
            printf("Tolerance for jacobi: %.6f\n", tol);
            tol = tolerance;
        }
    }


    // compute and time jacobi_offload_map on device with map
    if(offload_map) {
        if(debug_print) printf("Compute jacobi_offload_map on device with map\n");

        start_time = omp_get_wtime();
        if (tolerance >= 0) {
            iter = jacobi_offload_map_tol(f, u2, u_old2, N, iter_max, &tol);
        } else {
            time = jacobi_offload_map(f, u2, u_old2, N, iter_max);
        }
        end_time = omp_get_wtime();

        printf("Time for jacobi_offload_map: %.6f seconds\n", end_time - start_time);
        if (tolerance >= 0) {
            printf("Iter for jacobi_offload_map: %d\n", iter);
            printf("Tolerance for jacobi_offload_map: %.6f\n", tol);
            tol = tolerance;
        } else {
            printf("Time for jacobi_offload_map w/o transfer: %.6f seconds\n", time);
        }
        if(host) printf("Norm difference for jacobi_offload_map: %.6f\n", diff_norm_squared(u, u2, N));
    }


    // compute and time jacobi_offload on device
    if(offload) {
        if(debug_print) printf("Compute jacobi_offload on device\n");

        start_time = omp_get_wtime();
        if (tolerance >= 0) {
            iter = jacobi_offload_tol(f_d, u_d, u_old_d, N, iter_max, &tol);
        } else {
            jacobi_offload(f_d, u_d, u_old_d, N, iter_max);
        }
        end_time = omp_get_wtime();

        if(debug_print) printf("Copy device memory to host\n");
        omp_target_memcpy(u2[0][0], data_u_d, N * N * N * sizeof(double), 0, 0, host_num, dev_num);
        omp_target_memcpy(u_old2[0][0], data_u_old_d, N * N * N * sizeof(double), 0, 0, host_num, dev_num);

        printf("Time for jacobi_offload: %.6f seconds\n", end_time - start_time);
        if (tolerance >= 0) {
            printf("Iter for jacobi_offload: %d\n", iter);
            printf("Tolerance for jacobi_offload: %.6f\n", tol);
            tol = tolerance;
        }
        if(host) printf("Norm difference for jacobi_offload: %.6f\n", diff_norm_squared(u, u2, N));
    }


    // compute and time jacobi_offload_dual on two devices
    if(offload_dual) {
        // enable CUDA peer-to-peer access
        if(debug_print) printf("Turn on CUDA peer-to-peer access\n");
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0); // (dev 1, future flag)
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0); // (dev 0, future flag)
        cudaSetDevice(0);

        if(debug_print) printf("Compute jacobi_offload_dual on two devices\n");
        start_time = omp_get_wtime();
        if (tolerance >= 0) {
            iter = jacobi_offload_dual_tol(f_d0, f_d1, u_d0, u_d1, u_old_d0, u_old_d1, N, iter_max, &tol);
        } else {
            jacobi_offload_dual(f_d0, f_d1, u_d0, u_d1, u_old_d0, u_old_d1, N, iter_max);
        }
        end_time = omp_get_wtime();

        if(debug_print) printf("Copy device memory to host\n");
        omp_target_memcpy(u2[0][0], data_u_d0, N * N * N * sizeof(double) / 2, 0, 0, host_num, 0);
        omp_target_memcpy(u2[N/2][0], data_u_d1, N * N * N * sizeof(double) / 2, 0, 0, host_num, 1);
        omp_target_memcpy(u_old2[0][0], data_u_old_d0, N * N * N * sizeof(double) / 2, 0, 0, host_num, 0);
        omp_target_memcpy(u_old2[N/2][0], data_u_old_d1, N * N * N * sizeof(double) / 2, 0, 0, host_num, 1);

        printf("Time for jacobi_offload_dual: %.6f seconds\n", end_time - start_time);
        if (tolerance >= 0) {
            printf("Iter for jacobi_offload_dual: %d\n", iter);
            printf("Tolerance for jacobi_offload_dual: %.6f\n", tol);
            tol = tolerance;
        }
        if(host) printf("Norm difference for jacobi_offload_dual: %.6f\n", diff_norm_squared(u, u2, N));
    }
    

    // deallocate memory
    if(debug_print) printf("Deallocate memory on host\n");
    free_3d(f);
    free_3d(u);
    free_3d(u_old);
    free_3d(u2);
    free_3d(u_old2);


    // deallocate memory on device
    if(offload) {
        if(debug_print) printf("Deallocate memory on device\n");
        d_free_3d(f_d, data_f_d);
        d_free_3d(u_d, data_u_d);
        d_free_3d(u_old_d, data_u_old_d);
    }
    if(offload_dual) {
        if(debug_print) printf("Deallocate memory on the two devices\n");
        d_free_3d(f_d0, data_f_d0);
        d_free_3d(u_d0, data_u_d0);
        d_free_3d(u_old_d0, data_u_old_d0);
        d_free_3d(f_d1, data_f_d1);
        d_free_3d(u_d1, data_u_d1);
        d_free_3d(u_old_d1, data_u_old_d1);
    }

    return(0);
}

void init_f(double ***f, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                f[i][j][k] = 0.0;
            }
        }
    }

    // z-coordinate (modulo used as ceil)
    int i_start = (N - 1) / 6.0 + ((N - 1) % 6 != 0);   // (-2/3 + 1) * ((N-1) / 2) 
    int i_end = (N - 1) / 2.0;                          // (0    + 1) * ((N-1) / 2)

    // y-coordinate (not-flipped)
    int j_start = 0;                                    // (-1/2 + 1) * ((N-1) / 2)
    int j_end = (N - 1) / 4.0;                          // (-1   + 1) * ((N-1) / 2) 
 
    // x-coordinate
    int k_start = 0;                                    // (-1   + 1) * ((N-1) / 2) 
    int k_end = 5.0 * (N - 1) / 16.0;                   // (-3/8 + 1) * ((N-1) / 2) 

    for (int i = i_start; i <= i_end; i++) {
        for (int j = j_start; j <= j_end; j++) {
            for (int k = k_start; k <= k_end; k++) {
                f[i][j][k] = 200.0;
            }
        }
    }
}

void init_u_inner(double ***u, int N, double start_T) {
    #pragma omp parallel for
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                u[i][j][k] = start_T;
            }
        }
    }
}

void init_u_borders(double ***u, int N) {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                u[i][j][0] = 20.0;
                u[i][j][N-1] = 20.0;
            }
        }
        #pragma omp for nowait 
        for (int i = 1; i < N-1; i++) {
            for (int k = 1; k < N-1; k++) {
                u[i][0][k] = 0.0;
                u[i][N-1][k] = 20.0;
            }
        }
        #pragma omp for nowait 
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                u[0][j][k] = 20.0;
                u[N-1][j][k] = 20.0;
            }
        }
    }
}

double diff_norm_squared(double ***u1, double ***u2, int N) {
    double norm = 0.0;

    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                double x = u1[i][j][k] - u2[i][j][k];
                norm += x * x;
            }
        }
    }

    return norm;
}