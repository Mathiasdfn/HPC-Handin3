#ifndef _JACOBI_OFFLOAD_DUAL_H
#define _JACOBI_OFFLOAD_DUAL_H

void jacobi_offload_dual(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N, int iter_max);
void jacobi_offload_dual_border(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N, int iter_max);
int jacobi_offload_dual_tol(double ***f0, double ***f1, double ***u0, double ***u1, double ***u_old0, double ***u_old1, int N, int iter_max, double *tolerance);

#endif
