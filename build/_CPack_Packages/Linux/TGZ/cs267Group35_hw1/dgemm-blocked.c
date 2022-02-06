// #pragma GCC optimize("Ofast,inline") // Ofast = O3,fast-math,allow-store-data-races,no-protect-parens
// #pragma GCC target("avx,avx2,fma") // SIMD

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>
#include <avx2intrin.h>
const char* dgemm_desc = "Simple blocked dgemm.";
//#include "avxintrin-emu.h"

#ifndef L1_BLOCK
#define L2_BLOCK 1024
#define L1_BLOCK 256
#define REG_BLOCK 96
#endif


#define min(a,b) (((a)<(b))?(a):(b))


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

// static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
//     // For each row i of A
//     for (int i = 0; i < M; ++i) {
//         // For each column j of B
//         for (int j = 0; j < N; ++j) {
//             // Compute C(i,j)
//             double cij = C[i + j * lda];
//             for (int k = 0; k < K; ++k) {
//                 cij += A[i + k * lda] * B[k + j * lda];
//             }
//             C[i + j * lda] = cij;
//         }
//     }
// }

static inline void do_microkernel (int lda, int M, int N, int K, double* A, double* B, double* C)
{


  __m512d b_00, b_01, b_02, b_03;
  __m512d b_10, b_11, b_12, b_13;
  __m512d b_20, b_21, b_22, b_23;
  __m512d b_30, b_31, b_32, b_33;

  __m512d a_00,a_01,a_02,a_03;
  __m512d c_00,c_01,c_02,c_03;
  // __m512d a_X,c_X;

  /* For each column j of B */ 
  int i,j,k;
  for (j = 0; j < N; j+=4)
  { 
    for (k=0; k < K; k+=4)
    { // _mm512_set1_pd

      for (i = 0; i < M; i += 8)
      {

        b_00 = _mm512_set1_pd(*(B+k+j*lda));
        b_01 = _mm512_set1_pd(*(B+k+(j+1)*lda));
        b_02 = _mm512_set1_pd(*(B+k+(j+2)*lda));
        b_03 = _mm512_set1_pd(*(B+k+(j+3)*lda));
        b_10 = _mm512_set1_pd(*(B+k+1+j*lda));
        b_11 = _mm512_set1_pd(*(B+k+1+(j+1)*lda));
        b_12 = _mm512_set1_pd(*(B+k+1+(j+2)*lda));
        b_13 = _mm512_set1_pd(*(B+k+1+(j+3)*lda));
        b_20 = _mm512_set1_pd(*(B+k+2+j*lda));
        b_21 = _mm512_set1_pd(*(B+k+2+(j+1)*lda));
        b_22 = _mm512_set1_pd(*(B+k+2+(j+2)*lda));
        b_23 = _mm512_set1_pd(*(B+k+2+(j+3)*lda));
        b_30 = _mm512_set1_pd(*(B+k+3+j*lda));
        b_31 = _mm512_set1_pd(*(B+k+3+(j+1)*lda));
        b_32 = _mm512_set1_pd(*(B+k+3+(j+2)*lda));
        b_33 = _mm512_set1_pd(*(B+k+3+(j+3)*lda));


        a_00 = _mm512_load_pd(A + i+k*lda);
        a_01 = _mm512_load_pd(A + i+(k+1)*lda);
        a_02 = _mm512_load_pd(A + i+(k+2)*lda);
        a_03 = _mm512_load_pd(A + i+(k+3)*lda);


        c_00 = _mm512_load_pd(C + (i+j*lda));
        c_01 = _mm512_load_pd(C + (i+(j+1)*lda));
        c_02 = _mm512_load_pd(C + (i+(j+2)*lda));
        c_03 = _mm512_load_pd(C + (i+(j+3)*lda));


        c_00 = _mm512_add_pd(c_00, _mm512_mul_pd(a_00, b_00));
        c_01 = _mm512_add_pd(c_01, _mm512_mul_pd(a_00, b_01));
        c_02 = _mm512_add_pd(c_02, _mm512_mul_pd(a_00, b_02));
        c_03 = _mm512_add_pd(c_03, _mm512_mul_pd(a_00, b_03));

        c_00 = _mm512_add_pd(c_00, _mm512_mul_pd(a_01, b_10));
        c_01 = _mm512_add_pd(c_01, _mm512_mul_pd(a_01, b_11));
        c_02 = _mm512_add_pd(c_02, _mm512_mul_pd(a_01, b_12));
        c_03 = _mm512_add_pd(c_03, _mm512_mul_pd(a_01, b_13));

        c_00 = _mm512_add_pd(c_00, _mm512_mul_pd(a_02, b_20));
        c_01 = _mm512_add_pd(c_01, _mm512_mul_pd(a_02, b_21));
        c_02 = _mm512_add_pd(c_02, _mm512_mul_pd(a_02, b_22));
        c_03 = _mm512_add_pd(c_03, _mm512_mul_pd(a_02, b_23));

        c_00 = _mm512_add_pd(c_00, _mm512_mul_pd(a_03, b_30));
        c_01 = _mm512_add_pd(c_01, _mm512_mul_pd(a_03, b_31));
        c_02 = _mm512_add_pd(c_02, _mm512_mul_pd(a_03, b_32));
        c_03 = _mm512_add_pd(c_03, _mm512_mul_pd(a_03, b_33));
        /* Compute C(i,j) */
        _mm512_store_pd(C+i+j*lda, c_00);
        _mm512_store_pd(C+i+(j+1)*lda, c_01);
        _mm512_store_pd(C+i+(j+2)*lda, c_02);
        _mm512_store_pd(C+i+(j+3)*lda, c_03);

      }
    }
  }
}

static inline void L1_blocking_level (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* Accumulate block dgemms into block of C */
    /* For each block-row of A */
  for (int j = 0; j < N; j += REG_BLOCK)
  {
    /* For each block-column of B */
    for (int k = 0; k < K; k += REG_BLOCK) 
    {
      for (int i = 0; i < M; i += REG_BLOCK)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M1 = min (REG_BLOCK, M-i);
        int N1 = min (REG_BLOCK, N-j);
        int K1 = min (REG_BLOCK, K-k);

        /* Perform individual block dgemm */
        do_microkernel(lda, M1, N1, K1, A+i+k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}

static inline void L2_blocking_level (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* Accumulate block dgemms into block of C */
    /* For each block-row of A */
  for (int j = 0; j < N; j += L1_BLOCK)
  {
    /* For each block-column of B */
    for (int k = 0; k < K; k += L1_BLOCK)
    {
      for (int i = 0; i < M; i += L1_BLOCK)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M1 = min (L1_BLOCK, M-i);
        int N1 = min (L1_BLOCK, N-j);
        int K1 = min (L1_BLOCK, K-k);

        /* Perform individual block dgemm */
        L1_blocking_level(lda, M1, N1, K1, A+i+k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{

  // double * = _mm_malloc(sizeof(double)*lda*lda);

  // for (int i = 0; i < lda; ++i){
  //   for (int j = i+1; j < lda; ++j) {
  //       double temp = A[i*lda+j];
  //       A[i*lda+j] = A[j*lda+i];
  //       A[j*lda+i] = temp;
  //   }
  // }
  // for (int i = 0; i < N; i++)
  //     for (int j = 0; j < N; j++)
  //         B[i][j] = A[j][i];
  int powers_of_2[] = {2,4,8,16,32,64,128,256,512,1024};

  int lda_u = lda; 
  if (lda % 8){ 
      lda_u = lda + (8 - (lda % 8));
  } 

  if (lda_u == powers_of_2[4] || lda_u == powers_of_2[5] || lda_u == powers_of_2[6] || lda_u == powers_of_2[7] || lda_u == powers_of_2[8] || lda_u == powers_of_2[9]){
    lda_u += 8;
  }

  double *Apadded = _mm_malloc(sizeof(double)*lda_u*lda_u, 64); // certain instructions (SIMD) work on contiguous segments of 8 double words
  double *Bpadded = _mm_malloc(sizeof(double)*lda_u*lda_u, 64);
  double *Cpadded = _mm_malloc(sizeof(double)*lda_u*lda_u, 64);
  
  memset(Cpadded, 0.0, sizeof(double)*lda_u*lda_u);

  for (int i = 0; i < lda; ++i){
       memcpy(Apadded+i*lda_u, A+i*lda, sizeof(double)*lda);
       memcpy(Bpadded+i*lda_u, B+i*lda, sizeof(double)*lda);
       memset(Apadded+lda+i*lda_u, 0.0, sizeof(double)*(lda_u-lda));
       memset(Bpadded+lda+i*lda_u, 0.0, sizeof(double)*(lda_u-lda));
  }

  for (int i = lda; i < lda_u; ++i){
     memset(Apadded+i*lda_u, 0.0, sizeof(double)*lda_u);
     memset(Bpadded+i*lda_u, 0.0, sizeof(double)*lda_u);
  }


  /* For each block-column of B */
  for (int j = 0; j < lda_u; j += L2_BLOCK)
  {
    /* Accumulate block dgemms into block of C */
    for (int i = 0; i < lda_u; i += L2_BLOCK)
    {
      /* For each block-row of A */ 
      for (int k = 0; k < lda_u; k += L2_BLOCK)
      {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M = min (L2_BLOCK, lda_u-i);
          int N = min (L2_BLOCK, lda_u-j);
          int K = min (L2_BLOCK, lda_u-k);
          /* Perform individual block dgemm */
          //double Row_major_A[M * K];
          //transpose_row(Row_major_A,A,k*lda_u + i,K,M,lda_u);
          L2_blocking_level(lda_u, M, N, K, Apadded + k*lda_u + i, Bpadded+k+j*lda_u, Cpadded + i + j*lda_u);
      }
    }
  }

  for (int i = 0; i < lda; ++i)
  {
     memcpy(C+i*lda, Cpadded+i*lda_u, sizeof(double)*lda);
  }

  free(Cpadded);
  free(Bpadded);
  free(Apadded);

}