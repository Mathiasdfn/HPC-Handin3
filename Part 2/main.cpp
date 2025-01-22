/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "alloc3d.h"
#include "alloc3d_dev.h"
#include "jacobi.h"
#include "jacobi_offload.h"
#include "jacobi_offload_mem.h"

void init_f(double ***f, int N);
void init_u_inner(double ***u, int N, double start_T);
void init_u_borders(double ***u, int N);
void init_u_corners(double ***u, int N);
double diff_norm_squared(double ***u1, double ***u2, int N);
void print_3d_array(double ***A, int N);

int main(int argc, char *argv[]) {
    /* get the paramters from the command line */
    printf("Get parameters\n");
    int 	N = 100;
    int 	iter_max = 10000;
    double	start_T = 10;

    if (argc >= 2) N = atoi(argv[1]);	// grid size
    if (argc >= 3) iter_max = atoi(argv[2]);  // max. no. of iterations
    if (argc >= 4) start_T = atof(argv[3]);  // start T for all inner grid points

    // allocate memory
    printf("Allocate memory on host\n");
    double  ***f = NULL;
    double 	***u = NULL;
    double 	***u_old = NULL;
    double 	***u2 = NULL;
    double 	***u2_old = NULL;

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
    if ( (u2_old = malloc_3d(N, N, N)) == NULL ) {
        perror("array u_old: allocation failed");
        exit(-1);
    }

    // allocate memory on device
    printf("Allocate memory on device\n");
    double  ***f_d = NULL;
    double 	***u_d = NULL;
    double 	***u_old_d = NULL;
    double  **data_f_d = NULL;
    double 	**data_u_d = NULL;
    double 	**data_u_old_d = NULL;


    if ( (f_d = d_malloc_3d(N, N, N, data_f_d)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }
    if ( (u_d = d_malloc_3d(N, N, N, data_u_d)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (u_old_d = d_malloc_3d(N, N, N, data_u_old_d)) == NULL ) {
        perror("array u_old: allocation failed");
        exit(-1);
    }

    // initialize matrices
    printf("Initialize matrices on host\n");
    init_f(f, N);
    init_u_inner(u, N, start_T);
    init_u_borders(u, N);
    init_u_borders(u_old, N);
    init_u_inner(u2, N, start_T);
    init_u_borders(u2, N);
    init_u_borders(u2_old, N);

    // copy memory to device
    printf("Copy memory to device\n");
    int dev_num = omp_get_default_device();
    int host_num = omp_get_initial_device();
    omp_target_memcpy(f_d, f, N * N * N * sizeof(double), 0, 0, dev_num, host_num);
    omp_target_memcpy(u_d, u, N * N * N * sizeof(double), 0, 0, dev_num, host_num);
    omp_target_memcpy(u_old_d, u_old, N * N * N * sizeof(double), 0, 0, dev_num, host_num);

    // initialize debug NaN corners
    // init_u_corners(u, N);
    // init_u_corners(u_old, N);

    // compute and time jacobi with offload map
    printf("Compute jacobi on host\n");
    double  start_time, end_time, time;

    start_time = omp_get_wtime();
    jacobi(f, u, u_old, N, iter_max);
    end_time = omp_get_wtime();

    printf("Time for jacobi: %.6f seconds\n", end_time - start_time);

    printf("Compute jacobi on device with map\n");
    start_time = omp_get_wtime();
    time = jacobi_offload(f, u2, u2_old, N, iter_max);
    end_time = omp_get_wtime();

    printf("Time for jacobi_offload: %.6f seconds\n", end_time - start_time);
    printf("Time for jacobi_offload w/o transfer: %.6f seconds\n", time);
    printf("Norm difference ||u - u_d||^2: %.6f\n", diff_norm_squared(u, u2, N));

    // compute and time jacobi with offload target memory
    printf("Compute jacobi on device with target memory\n");
    start_time = omp_get_wtime();
    jacobi_offload_mem(f_d, u_d, u_old_d, N, iter_max);
    end_time = omp_get_wtime();

    omp_target_memcpy(u2, u_d, N * N * N * sizeof(double), 0, 0, host_num, dev_num);
    omp_target_memcpy(u2_old, u_old_d, N * N * N * sizeof(double), 0, 0, host_num, dev_num);

    printf("Time for jacobi_offload_mem: %.6f seconds\n", end_time - start_time);
    printf("Norm difference ||u - u_d||^2: %.6f\n", diff_norm_squared(u, u2, N));
    
    // de-allocate memory
    printf("Deallocate memory on host\n");
    free_3d(f);
    free_3d(u);
    free_3d(u_old);
    free_3d(u2);
    free_3d(u2_old);

    // de-allocate memory on device
    printf("Deallocate memory on host\n");
    d_free_3d(f_d, *data_f_d);
    d_free_3d(u_d, *data_u_d);
    d_free_3d(u_old_d, *data_u_old_d);

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

void init_u_corners(double ***u, int N) {
    u[0][0][0] = NAN;
    u[0][0][N-1] = NAN;
    u[0][N-1][0] = NAN;
    u[0][N-1][N-1] = NAN;
    u[N-1][0][0] = NAN;
    u[N-1][0][N-1] = NAN;
    u[N-1][N-1][0] = NAN;
    u[N-1][N-1][N-1] = NAN;
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

void print_3d_array(double ***A, int N) {
    for (int i = 0; i < N; i++) {
        printf("Slice i = %d:\n", i);
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                printf("%.2f ", A[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}