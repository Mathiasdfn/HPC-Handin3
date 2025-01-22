/* jacobi.c - Poisson problem in 3d
 * 
 */

void swap_pointers(void **ptr1, void **ptr2);
void update_u(double ***f, double ***u, double ***u_old, int N);

void jacobi(double ***f, double ***u, double ***u_old, int N, int iter_max) {
    for(int iter=0; iter < iter_max; iter++) {
        swap_pointers((void **)&u, (void **)&u_old);
        update_u(f, u, u_old, N);
    }
}

void swap_pointers(void **ptr1, void **ptr2) {
    void *temp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = temp;
}

void update_u(double ***f, double ***u, double ***u_old, int N) {
    double delta = 2.0 / (N - 1);
    double delta2 = delta * delta;
    double frac = 1.0 / 6.0;

    #pragma omp parallel for schedule(static)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                u[i][j][k] = frac * (u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] + delta2*f[i][j][k]);
            }
        }
    }
}
