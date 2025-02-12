/* jacobi_offload_map.c - Poisson problem in 3d offloaded with map
 * 
 */
#include <omp.h>
#include <math.h>

void swap_pointers(void **ptr1, void **ptr2);
void update_u_offload_map(double ***f, double ***u, double ***u_old, int N);
double update_u_with_diff_norm_squared_offload_map(double ***f, double ***u, double ***u_old, int N);

double jacobi_offload_map(double ***f, double ***u, double ***u_old, int N, int iter_max) {
    double start_time, end_time;

    #pragma omp target data \
        map(to: f[0:N][0:N][0:N]) \
        map(tofrom: u[0:N][0:N][0:N], u_old[0:N][0:N][0:N])
    {
        start_time = omp_get_wtime();
        for(int iter=0; iter < iter_max; iter++) {
            swap_pointers((void **)&u, (void **)&u_old);
            update_u_offload_map(f, u, u_old, N);
        }
        end_time = omp_get_wtime();
    }

    return end_time - start_time;
}

int jacobi_offload_map_tol(double ***f, double ***u, double ***u_old, int N, int iter_max, double *tolerance) {
    int iter = 0;
    double d2 = INFINITY;
    double tol = *tolerance;
    double tol2 = tol * tol;

    #pragma omp target data \
        map(to: f[0:N][0:N][0:N]) \
        map(tofrom: u[0:N][0:N][0:N], u_old[0:N][0:N][0:N])
    while (d2 > tol2 && iter < iter_max) {
        swap_pointers((void **)&u, (void **)&u_old);
        d2 = update_u_with_diff_norm_squared_offload_map(f, u, u_old, N);
        iter++;
    }

    *tolerance = sqrt(d2);
    return iter;
}

void update_u_offload_map(double ***f, double ***u, double ***u_old, int N) {
    double delta = 2.0 / (N - 1);
    double delta2 = delta * delta;
    double frac = 1.0 / 6.0;

    #pragma omp target teams distribute parallel for collapse(3)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                u[i][j][k] = frac * (u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] 
                    + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] + delta2*f[i][j][k]);
            }
        }
    }
}

double update_u_with_diff_norm_squared_offload_map(double ***f, double ***u, double ***u_old, int N) {
    double delta = 2.0 / (N - 1);
    double delta2 = delta * delta;
    double frac = 1.0 / 6.0;
    double norm = 0.0;

    #pragma omp target teams distribute parallel for collapse(3) reduction(+: norm)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                double u_ijk = frac * (u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] + delta2*f[i][j][k]);
                double x = u_ijk - u_old[i][j][k];
                norm += x * x;
                u[i][j][k] = u_ijk;
            }
        }
    }

    return norm;
}
