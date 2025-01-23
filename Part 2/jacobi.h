#ifndef _JACOBI_H
#define _JACOBI_H

void jacobi(double ***f, double ***u, double ***u_old, int N, int iter_max);
int jacobi_tol(double ***f, double ***u, double ***u_old, int N, int iter_max, double *tolerance);

#endif
