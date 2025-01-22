/* jacobi_offload.c - Poisson problem in 3d offloaded
 * 
 */
#include <omp.h>

void swap_pointers_offload_dual(void **ptr1, void **ptr2);
void update_u_offload_dual(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N);

void jacobi_offload_dual(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N, int iter_max) {
    for(int iter=0; iter < iter_max; iter++) {
        swap_pointers_offload_dual((void **)&u0, (void **)&u_old0);
        swap_pointers_offload_dual((void **)&u1, (void **)&u_old1);
        update_u_offload_dual(f0, f1, u0, u1, u_old0, u_old1, N);
    }
}

void swap_pointers_offload_dual(void **ptr1, void **ptr2) {
    void *temp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = temp;
}   

void update_u_offload_dual(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N) {
    double delta = 2.0 / (N - 1);
    double delta2 = delta * delta;
    double frac = 1.0 / 6.0;

    // inner points
    #pragma omp target teams loop collapse(3) device(0) is_device_ptr(f0, u0, u_old0) nowait
    for (int i = 1; i < N/2 - 1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                u0[i][j][k] = frac * (u_old0[i-1][j][k] + u_old0[i+1][j][k] + u_old0[i][j-1][k] 
                    + u_old0[i][j+1][k] + u_old0[i][j][k-1] + u_old0[i][j][k+1] + delta2*f0[i][j][k]);
            }
        }
    }

    #pragma omp target teams loop collapse(3) device(1) is_device_ptr(f1, u1, u_old1) nowait
    for (int i = 1; i < N/2 - 1; i++) {
        for (int j = 1; j < N-1; j++) {
            for (int k = 1; k < N-1; k++) {
                u1[i][j][k] = frac * (u_old1[i-1][j][k] + u_old1[i+1][j][k] + u_old1[i][j-1][k] 
                    + u_old1[i][j+1][k] + u_old1[i][j][k-1] + u_old1[i][j][k+1] + delta2*f1[i][j][k]);
            }
        }
    }

    // border mid points
    int i0 = N/2 - 1;
    int i1 = 0;

    #pragma omp target teams loop collapse(2) device(0) is_device_ptr(f0, u0, u_old0, u_old1) nowait
    for (int j = 1; j < N-1; j++) {
        for (int k = 1; k < N-1; k++) {
            u0[i0][j][k] = frac * (u_old0[i0-1][j][k] + u_old1[i1][j][k] + u_old0[i0][j-1][k] 
                + u_old0[i0][j+1][k] + u_old0[i0][j][k-1] + u_old0[i0][j][k+1] + delta2*f0[i0][j][k]);
        }
    }


    #pragma omp target teams loop collapse(2) device(1) is_device_ptr(f1, u1, u_old1, u_old0) nowait
    for (int j = 1; j < N-1; j++) {
        for (int k = 1; k < N-1; k++) {
            u1[i1][j][k] = frac * (u_old0[i0][j][k] + u_old1[i1+1][j][k] + u_old1[i1][j-1][k] 
                + u_old1[i1][j+1][k] + u_old1[i1][j][k-1] + u_old1[i1][j][k+1] + delta2*f1[i1][j][k]);
        }
    }

    #pragma omp taskwait
}