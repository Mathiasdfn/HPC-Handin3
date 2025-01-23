/* jacobi_offload.c - Poisson problem in 3d offloaded
 * 
 */
#include <omp.h>
#include <math.h>

void swap_pointers(void **ptr1, void **ptr2);
void update_u_offload_dual(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N);
double update_u_with_diff_norm_squared_offload_dual(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N);

void jacobi_offload_dual(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N, int iter_max) {
    for(int iter=0; iter < iter_max; iter++) {
        swap_pointers((void **)&u0, (void **)&u_old0);
        swap_pointers((void **)&u1, (void **)&u_old1);
        update_u_offload_dual(f0, f1, u0, u1, u_old0, u_old1, N);
    }
}

int jacobi_offload_dual_tol(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N, int iter_max, double *tolerance) {
    int iter = 0;
    double d2 = INFINITY;
    double tol = *tolerance;
    double tol2 = tol * tol;

    while (d2 > tol2 && iter < iter_max) {
        swap_pointers((void **)&u0, (void **)&u_old0);
        swap_pointers((void **)&u1, (void **)&u_old1);
        d2 = update_u_with_diff_norm_squared_offload_dual(f0, f1, u0, u1, u_old0, u_old1, N);
        iter++;
    }

    *tolerance = sqrt(d2);
    return iter;
}

void update_u_offload_dual(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N) {
    double delta = 2.0 / (N - 1);
    double delta2 = delta * delta;
    double frac = 1.0 / 6.0;
    int i0 = N/2 - 1;
    int i1 = 0;

    #pragma omp target teams loop collapse(3) \
        device(0) is_device_ptr(f0, u0, u_old0, u_old1) nowait
    for (int i = 1; i < N/2; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                double u_old_i1jk = i == i0 ? u_old1[i1][j][k] : u_old0[i+1][j][k];

                u0[i][j][k] = frac * (u_old0[i-1][j][k] + u_old_i1jk + u_old0[i][j-1][k] 
                    + u_old0[i][j+1][k] + u_old0[i][j][k-1] + u_old0[i][j][k+1] + delta2*f0[i][j][k]);
            }
        }
    }

    #pragma omp target teams loop collapse(3) \
        device(1) is_device_ptr(f1, u1, u_old1, u_old0) nowait
    for (int i = 0; i < N/2 - 1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                double u_old_i1jk = i == i1 ? u_old0[i0][j][k] : u_old1[i-1][j][k];
                
                u1[i][j][k] = frac * (u_old_i1jk + u_old1[i+1][j][k] + u_old1[i][j-1][k] 
                    + u_old1[i][j+1][k] + u_old1[i][j][k-1] + u_old1[i][j][k+1] + delta2*f1[i][j][k]);
            }
        }
    }

    #pragma omp taskwait
}

double update_u_with_diff_norm_squared_offload_dual(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N) {
    double delta = 2.0 / (N - 1);
    double delta2 = delta * delta;
    double frac = 1.0 / 6.0;
    int i0 = N/2 - 1;
    int i1 = 0;
    double norm0 = 0.0;
    double norm1 = 0.0;

    #pragma omp target teams loop collapse(3) reduction(+: norm0) \
        device(0) is_device_ptr(f0, u0, u_old0, u_old1) nowait
    for (int i = 1; i < N/2; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                double u_old_i1jk = i == i0 ? u_old1[i1][j][k] : u_old0[i+1][j][k];
                double u0_ijk = frac * (u_old0[i-1][j][k] + u_old_i1jk + u_old0[i][j-1][k] 
                    + u_old0[i][j+1][k] + u_old0[i][j][k-1] + u_old0[i][j][k+1] + delta2*f0[i][j][k]);

                double x = u0_ijk - u_old0[i][j][k];
                norm0 += x * x;
                u0[i][j][k] = u0_ijk;
            }
        }
    }

    #pragma omp target teams loop collapse(3) reduction(+: norm1) \
        device(1) is_device_ptr(f1, u1, u_old1, u_old0) nowait
    for (int i = 0; i < N/2 - 1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                double u_old_i1jk = i == i1 ? u_old0[i0][j][k] : u_old1[i-1][j][k];
                double u1_ijk = frac * (u_old_i1jk + u_old1[i+1][j][k] + u_old1[i][j-1][k] 
                    + u_old1[i][j+1][k] + u_old1[i][j][k-1] + u_old1[i][j][k+1] + delta2*f1[i][j][k]);

                double x = u1_ijk - u_old1[i][j][k];
                norm1 += x * x;
                u1[i][j][k] = u1_ijk;
            }
        }
    }

    #pragma omp taskwait
    return norm0 + norm1;
}