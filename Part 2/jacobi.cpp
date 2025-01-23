/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>

void swap_pointers(void **ptr1, void **ptr2);
void update_u(double ***f, double ***u, double ***u_old, int N);
double update_u_with_diff_norm_squared(double ***f, double ***u, double ***u_old, int N);

void jacobi(double ***f, double ***u, double ***u_old, int N, int iter_max) {
    for(int iter=0; iter < iter_max; iter++) {
        swap_pointers((void **)&u, (void **)&u_old);
        update_u(f, u, u_old, N);
    }
}

int jacobi_tol(double ***f, double ***u, double ***u_old, int N, int iter_max, double *tolerance) {
    int iter = 0;
    double d2 = INFINITY;
    double tol = *tolerance;
    double tol2 = tol * tol;

    while (d2 > tol2 && iter < iter_max) {
        swap_pointers((void **)&u, (void **)&u_old);
        d2 = update_u_with_diff_norm_squared(f, u, u_old, N);
        iter++;
    }

    *tolerance = sqrt(d2);
    return iter;
}

void update_u(double ***f, double ***u, double ***u_old, int N) {
    double delta = 2.0 / (N - 1);
    double delta2 = delta * delta;
    double frac = 1.0 / 6.0;

    #pragma omp parallel for
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                u[i][j][k] = frac * (u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] + delta2*f[i][j][k]);
            }
        }
    }
}

double update_u_with_diff_norm_squared(double ***f, double ***u, double ***u_old, int N) {
    double delta = 2.0 / (N - 1);
    double delta2 = delta * delta;
    double frac = 1.0 / 6.0;
    double norm = 0.0;

    #pragma omp parallel for reduction(+: norm)
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
