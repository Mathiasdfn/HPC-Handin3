#ifndef _JACOBI_OFFLOAD_H
#define _JACOBI_OFFLOAD_H

void jacobi_offload(double ***f, double ***u, double ***u_old, int N, int iter_max) ;
int jacobi_offload_tol(double ***f, double ***u, double ***u_old, int N, int iter_max, double *tolerance);

#endif
