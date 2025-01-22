// Load libraries
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
extern "C" {
#include <cblas.h>
void matmult_nat(int m, int n, int k, double **A, double **B, double **C);
void matmult_lib(int m, int n, int k, double **A, double **B, double **C);
void matmult_mnk(int m, int n, int k, double **A, double **B, double **C);
void matmult_nmk(int m, int n, int k, double **A, double **B, double **C);
void matmult_mkn(int m, int n, int k, double **A, double **B, double **C);
void matmult_nkm(int m, int n, int k, double **A, double **B, double **C);
void matmult_kmn(int m, int n, int k, double **A, double **B, double **C);
void matmult_knm(int m, int n, int k, double **A, double **B, double **C);
void matmult_blk(int m, int n, int k, double **A, double **B, double **C, int bs);
void matmult_mkn_omp(int m, int n, int k, double **A, double **B, double **C);
void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double **C);
void matmult_mnk_offload(int m, int n, int k, double **A, double **B, double **C);
void matmult_blk_offload(int m, int n, int k, double **A, double **B, double **C, int bs);
void matmult_asy_offload(int m, int n, int k, double **A, double **B, double **C);
void matmult_lib_offload(int m, int n, int k, double **A, double **B, double **C);
#define MIN(a, b) ((a) < (b) ? (a) : (b))
}


// #define MIN(a, b) ((a) < (b) ? (a) : (b))

// ################# ASSIGNMENT ##########################


// PART I
void matmult_nat(int m, int n, int k, double **A, double **B, double **C){
    for (int i = 0; i < m ; i++){          // Loop over rows in A
        for (int j = 0; j < n; j++){       // Loop over collumns in B
            C[i][j] = 0;
            for (int l=0 ; l < k; l++){    // Loop over the cols in A and rows in B
                C[i][j] += A[i][l]*B[l][j];
    }}}    
}

void matmult_lib(int m, int n, int k, double **A, double **B, double **C){
    int alpha = 1, beta = 0;
    int lda = k, ldb = n, ldc = n;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A[0], lda, B[0], ldb, beta, C[0], ldc);
}

// PART II
void matmult_mnk(int m, int n, int k, double **A, double **B, double **C){
    matmult_nat(m, n, k, A, B, C);
}

void matmult_nmk(int m, int n, int k, double **A, double **B, double **C){
    for (int j = 0; j < n; j++){                // Loop over collumns in B    
        for (int i = 0; i < m ; i++){           // Loop over rows in A
            C[i][j] = 0;
            for (int l=0 ; l < k; l++){         // Loop over the cols in A and rows in B
                C[i][j] += A[i][l]*B[l][j];
    }}}  
}

void matmult_mkn(int m, int n, int k, double **A, double **B, double **C){
    // Set C matrix to 0
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            C[i][j] = 0;
    }}
    
    // Compute A*B product
    for (int i=0; i<m; i++){
        for (int l=0; l<k; l++){
            for (int j=0; j<n; j++){
                C[i][j] += A[i][l]*B[l][j];
    }}}
}


void matmult_nkm(int m, int n, int k, double **A, double **B, double **C){
     // Set C matrix to 0
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            C[i][j] = 0;
        }
    }

    // Compute A*B product
    for (int j=0; j<n; j++){
        for (int l=0; l<k; l++){
            for (int i=0; i<m; i++){
                C[i][j] += A[i][l]*B[l][j];
    }}}
}

void matmult_kmn(int m, int n, int k, double **A, double **B, double **C){
    // Set C matrix to 0
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            C[i][j] = 0;
    }}

    // Compute A*B product
    for (int l=0; l<k; l++){
        for (int i=0; i<m; i++){
            for (int j=0; j<n; j++){
                C[i][j] += A[i][l]*B[l][j];
    }}}
}

void matmult_knm(int m, int n, int k, double **A, double **B, double **C){
    // Set C matrix to 0
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            C[i][j] = 0;
        }
    }

    // Compute A*B product
    for (int l=0; l<k; l++){
        for (int j=0; j<n; j++){
            for (int i=0; i<m; i++){
                C[i][j] += A[i][l]*B[l][j];
    }}}
}


// Part IV
void matmult_blk(int m, int n, int k, double **A, double **B, double **C, int bs) {
    // Set C matrix to 0
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
        }
    }

    // Compute the blocked A*B product
    for (int ib = 0; ib < m; ib += bs) {
        for (int lb = 0; lb < k; lb += bs) {
            for (int jb = 0; jb < n; jb += bs) {
                // Ensure we do not go out of bounds
                int ie = MIN(ib + bs, m);
                int le = MIN(lb + bs, k);
                int je = MIN(jb + bs, n);

                for (int i = ib; i < ie; i++) {
                    for (int l = lb; l < le; l++) {
                        for (int j = jb; j < je; j++) {
                            C[i][j] += A[i][l] * B[l][j];
    }}}}}}
}

// 3.1
void matmult_mkn_omp(int m, int n, int k, double **A, double **B, double **C) {
    // Set C matrix to 0
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            C[i][j] = 0;
    }}
    
    // Compute A*B product
    #pragma omp parallel for
    for (int i=0; i<m; i++){
        for (int l=0; l<k; l++){
            for (int j=0; j<n; j++){
                C[i][j] += A[i][l]*B[l][j];
    }}}
}

// 3.2
void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double **C) {
    // Set C matrix to 0
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            C[i][j] = 0;
    }}
    
    // Compute A*B product
    #pragma omp target teams map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n]) \
            num_teams(m) thread_limit(32)
    #pragma omp distribute collapse(2)
    for (int i=0; i<m; i++){
        for (int l=0; l<k; l++){
            #pragma omp parallel for
            for (int j=0; j<n; j++){
                C[i][j] += A[i][l]*B[l][j];
            }
        }
    }
}

void matmult_mnk_offload(int m, int n, int k, double **A, double **B, double **C) {
    // No need for init as C is set in the following loop

    // Compute A*B product
    #pragma omp target teams map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n]) \
            num_teams(m) thread_limit(32)
    #pragma omp distribute collapse(2)
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            double sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int l=0; l<k; l++){
                sum += A[i][l]*B[l][j];
            }
            C[i][j] = sum;
        }
    }
}

// 3.3
void matmult_blk_offload(int m, int n, int k, double **A, double **B, double **C, int bs){
    #define BLK 5 // Define block size at compile time

    // Compute the blocked A*B product
    #pragma omp target teams distribute parallel for \
            num_teams(114) thread_limit(64)\
            map(to: A[0:m][0:k], B[0:k][0:n]) map(from: C[0:m][0:n])

    for (int ib = 0; ib < m; ib += BLK) {
        for (int j = 0; j < n; j++) {
            if (ib + BLK - 1 < m){ // Ensure we do not go out of bounds
                double sum[BLK] = {0};
                for (int l = 0; l < k; l++) {
                    for (int i = 0; i < BLK; i++) {
                        sum[i] += A[i+ib][l] * B[l][j];
                }}

                for (int i = 0; i < BLK; i++) {
                    C[i+ib][j] = sum[i];
                }
            } else {
                double sum[BLK] = {0};
                for (int l = 0; l < k; l++) {
                    for (int i = 0; i < m-ib; i++) {
                    sum[i] += A[i + ib][l] * B[l][j];
                }}

                for (int i = 0; i < m-ib; i++) {
                    C[i+ib][j] = sum[i];
                }
            }
        } 
    }
}


void matmult_asy_offload(int m, int n, int k, double **A, double **B, double **C){
    //#pragma target data map(to:A[0:m][0:k], B[0:k][0:n]) map(from: C[0:m][0:n])
    {
        #define SLAPS 2
      
        for (int s = 0; s < SLAPS; s++) {
            
            int length = m/SLAPS;
            int slap_start = s*length;

            //#pragma omp target update to(A[slap_start+i:length][0:n])

            #pragma omp target teams distribute parallel for collapse(2) nowait\
                    num_teams(length) thread_limit(n*k/4)\
                    map(to:A[slap_start:length][0:k], B[0:k][0:n]) map(from:C[slap_start:length][0:n])

                for (int i = slap_start; i < length+slap_start; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0 ;
                        for (int l = 0; l < k; l++) {
                            sum += A[i][l] * B[l][j];
                        }
                        C[i][j] = sum;
                    }
                //#pragma omp target update from(C[slap_start+i:length][0:n])
                }
            }
            #pragma omp taskwait
    } /* Exit data region*/
}