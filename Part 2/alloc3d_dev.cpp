#include <omp.h>
#include <stdio.h>

double ***d_malloc_3d(int m, int n, int k, double **data) {
    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    int dev_num = omp_get_default_device();

    double ***p = (double***) omp_target_alloc(m * sizeof(double **) + m * n * sizeof(double *), dev_num);
    if (p == NULL) {
        return NULL;
    }

    #pragma omp target is_device_ptr(p)
    for(int i = 0; i < m; i++) {
        p[i] = (double **) p + m + i * n;
    }

    double *a = (double*) omp_target_alloc(m * n * k * sizeof(double), dev_num);
    if (a == NULL) {
	    omp_target_free(p, dev_num);
	    return NULL;
    }

    #pragma omp target is_device_ptr(p, a)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }

    *data = a;

    return p;
}

void d_free_3d(double ***p, double *data) {
    int dev_num = omp_get_default_device();
    omp_target_free(data, dev_num);
    omp_target_free(p, dev_num);
}
