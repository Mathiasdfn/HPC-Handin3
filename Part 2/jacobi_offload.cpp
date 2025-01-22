/* jacobi_offload.c - Poisson problem in 3d offloaded
 * 
 */
#include <omp.h>

void swap_pointers_offload(void **ptr1, void **ptr2);
void update_u_offload(double ***f, double ***u, double ***u_old, int N);

void jacobi_offload(double ***f, double ***u, double ***u_old, int N, int iter_max) {
    for(int iter=0; iter < iter_max; iter++) {
        swap_pointers_offload((void **)&u, (void **)&u_old);
        update_u_offload(f, u, u_old, N);
    }
}

void swap_pointers_offload(void **ptr1, void **ptr2) {
    void *temp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = temp;
}

void update_u_offload(double ***f, double ***u, double ***u_old, int N) {
    double delta = 2.0 / (N - 1);
    double delta2 = delta * delta;
    double frac = 1.0 / 6.0;

    #pragma omp target teams loop collapse(3) is_device_ptr(f, u, u_old)
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                u[i][j][k] = frac * (u_old[i-1][j][k] + u_old[i+1][j][k] + u_old[i][j-1][k] 
                    + u_old[i][j+1][k] + u_old[i][j][k-1] + u_old[i][j][k+1] + delta2*f[i][j][k]);
            }
        }
    }
}