#ifndef __ALLOC_3D_DEV
#define __ALLOC_3D_DEV

double ***d_malloc_3d_dev(int m, int n, int k);

#define HAS_FREE_3D_DEV
void d_free_3d_dev(double ***array3D);

#endif /* __ALLOC_3D_DEV */
