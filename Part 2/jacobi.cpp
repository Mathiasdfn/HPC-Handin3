/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>

void swap_pointers(void **ptr1, void **ptr2);
double update_u_with_l2norm2(double ***f, double ***u, double ***u_old, int N);

int jacobi(double ***f, double ***u, double ***u_old, int N, int iter_max) {
    int iter = 0;
    // double d2 = INFINITY;
    // double tol = *tolerance;
    // double tol2 = tol * tol;

    while (iter < iter_max) {
        swap_pointers((void **)&u, (void **)&u_old);
        // d2 = update_u_with_l2norm2(f, u, u_old, N);
        iter++;
    }

    // *tolerance = sqrt(d2);
    return iter;
}

void swap_pointers(void **ptr1, void **ptr2) {
    void *temp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = temp;
}

double update_u_with_l2norm2(double ***f, double ***u, double ***u_old, int N) {
    double delta = 2.0 / (N - 1);
    double delta2 = delta * delta;
    double frac = 1.0 / 6.0;
    double sum = 0.0;

    #pragma omp parallel for schedule(static) reduction(+: sum)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                double u_ijk = frac * (u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] + delta2*f[i][j][k]);
                double x = u_ijk - u_old[i][j][k];
                sum += x * x;
                u[i][j][k] = u_ijk;
            }
        }
    }

    return sum;
}