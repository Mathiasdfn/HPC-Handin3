#ifndef _JACOBI_OFFLOAD_MAP_H
#define _JACOBI_OFFLOAD_MAP_H

double jacobi_offload_map(double ***f, double ***u, double ***u_old, int N, int iter_max);
int jacobi_offload_map_tol(double ***f, double ***u, double ***u_old, int N, int iter_max, double *tolerance);

#endif
