/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "alloc3d.h"
#include "print.h"
#include "jacobi.h"


#define N_DEFAULT 100

void init_f(double ***f, int N);
void init_u_inner(double ***u, int N, double start_T);
void init_u_borders(double ***u, int N);
void init_u_corners(double ***u, int N);

int main(int argc, char *argv[]) {
    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    // double	tolerance;
    double	start_T;

    double  ***f = NULL;
    double 	***u = NULL;
    double 	***u_old = NULL;

    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    // tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[3]);  // start T for all inner grid points

    // allocate memory
    if ( (f = malloc_3d(N, N, N)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }
    if ( (u = malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (u_old = malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    // initialize matrices
    init_f(f, N);
    init_u_inner(u, N, start_T);
    init_u_borders(u, N);
    init_u_borders(u_old, N);

    // initialize debug NaN corners
    // init_u_corners(u, N);
    // init_u_corners(u_old, N);

    // compute Jacobi method

    int iter = jacobi(f, u, u_old, N, iter_max);
    

    // de-allocate memory
    free_3d(f);
    free_3d(u);
    free_3d(u_old);

    return(0);
}

void init_f(double ***f, int N) {
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
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                u[i][j][k] = start_T;
            }
        }
    }
}

void init_u_borders(double ***u, int N) {
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            u[i][j][0] = 20.0;
            u[i][j][N-1] = 20.0;
        }
    }
    for (int i = 1; i < N-1; i++) {
        for (int k = 1; k < N-1; k++) {
            u[i][0][k] = 0.0;
            u[i][N-1][k] = 20.0;
        }
    }
    for (int j = 1; j < N-1; j++) {
        for (int k = 1; k < N-1; k++) {
            u[0][j][k] = 20.0;
            u[N-1][j][k] = 20.0;
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
