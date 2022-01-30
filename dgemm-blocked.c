#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>
#include <avx2intrin.h>
const char* dgemm_desc = "Simple blocked dgemm.";
//#include "avxintrin-emu.h"

#ifndef L1_BLOCK
#define L3_BLOCK 1024
#define L2_BLOCK 256
#define L1_BLOCK 128
#endif


#define min(a,b) (((a)<(b))?(a):(b))


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */


static inline void do_block_trial (int lda, int M, int N, int K, double* A, double* B, double* C)
{


  __m256d b_00, b_10, b_01, b_11, b_02, b_12, b_03, b_13, b_20, b_21, b_22, b_23, b_30, b_31, b_32, b_33;
  __m256d a_00,a_01,a_02,a_03,a_10, a_11, a_12, a_13, a_20, a_21, a_22, a_23, a_30, a_31, a_32, a_33;
  __m256d c_00,c_01,c_02,c_03,c_10, c_11, c_12, c_13, c_20, c_21, c_22, c_23, c_30, c_31, c_32, c_33;
  __m256d a_X,c_X;

  /* For each row i of A */
  for (int j = 0; j < N; j+=4)
  {
  /* For each column j of B */ 
    for (int k=0; k < K; k+=4){

      b_00 = _mm256_broadcast_sd(B+k+j*lda);
      b_10 = _mm256_broadcast_sd(B+k+(j+1)*lda);
      b_20 = _mm256_broadcast_sd(B+k+(j+2)*lda);
      b_30 = _mm256_broadcast_sd(B+k+(j+3)*lda);
      b_01 = _mm256_broadcast_sd(B+k+1+j*lda);
      b_11 = _mm256_broadcast_sd(B+k+1+(j+1)*lda);
      b_21 = _mm256_broadcast_sd(B+k+1+(j+2)*lda);
      b_31 = _mm256_broadcast_sd(B+k+1+(j+3)*lda);
      b_02 = _mm256_broadcast_sd(B+k+2+j*lda);
      b_12 = _mm256_broadcast_sd(B+k+2+(j+1)*lda);
      b_22 = _mm256_broadcast_sd(B+k+2+(j+2)*lda);
      b_32 = _mm256_broadcast_sd(B+k+2+(j+3)*lda);
      b_03 = _mm256_broadcast_sd(B+k+3+j*lda);
      b_13 = _mm256_broadcast_sd(B+k+3+(j+1)*lda);
      b_23 = _mm256_broadcast_sd(B+k+3+(j+2)*lda);
      b_33 = _mm256_broadcast_sd(B+k+3+(j+3)*lda);



      for (int i = 0; i < M; i += 8)
      {


        /* Compute C(i,j) */
        
        //double A_row_major[4] __attribute__ ((aligned (32)));
        // A_row_major[0] = A[i+k*lda];
        // A_row_major[1] = A[i+(k+1)*lda];
        // A_row_major[2] = A[i+(k+2)*lda];
        // A_row_major[3] = A[i+(k+3)*lda];
        //a_X = _mm256_load_pd(A_row_major);

        
        
        //c_0 = _mm256_fmadd_pd(a_X,b_00,c_0);
        //c_1 = _mm256_fmadd_pd(a_X,b_10,c_1);
        // A_row_major[0] = *(A + i+k*lda);
        // A_row_major[1] = *(A + i+(k+1)*lda);
        // A_row_major[2] = *(A + i+(k+2)*lda);
        // A_row_major[3] = *(A + i+(k+3)*lda);
        // a_X = _mm256_load_pd(A_row_major);
        // //c_X = _mm256_add_pd(c_0, _mm256_mul_pd(a_X, b_00));
        
        
        // A[i+1+k*lda] = A[i+(k+1)*lda];
        // A[i+2+k*lda] = A[i+(k+2)*lda];
        // A[i+3+k*lda] = A[i+(k+3)*lda];
        a_00 = _mm256_load_pd(A + i+k*lda);
        a_01 = _mm256_load_pd(A + i+(k+1)*lda);
        a_02 = _mm256_load_pd(A + i+(k+2)*lda);
        a_03 = _mm256_load_pd(A + i+(k+3)*lda);
        a_10 = _mm256_load_pd(A + i+4+k*lda);
        a_11 = _mm256_load_pd(A + i+4+(k+1)*lda);
        a_12 = _mm256_load_pd(A + i+4+(k+2)*lda);
        a_13 = _mm256_load_pd(A + i+4+(k+3)*lda);
        a_20 = _mm256_load_pd(A + i+8+k*lda);
        a_21 = _mm256_load_pd(A + i+8+(k+1)*lda);
        a_22 = _mm256_load_pd(A + i+8+(k+2)*lda);
        a_23 = _mm256_load_pd(A + i+8+(k+3)*lda);
        a_30 = _mm256_load_pd(A + i+12+k*lda);
        a_31 = _mm256_load_pd(A + i+12+(k+1)*lda);
        a_32 = _mm256_load_pd(A + i+12+(k+2)*lda);
        a_33 = _mm256_load_pd(A + i+12+(k+3)*lda);
        // A[i+(k+1)*lda] =  A[i+1+k*lda];
        // A[i+(k+2)*lda] =  A[i+2+k*lda];
        // A[i+(k+3)*lda] =  A[i+3+k*lda];
        // c_X = _mm256_fmadd_pd(a_0,b_00,c_0);
        c_00 = _mm256_load_pd(C + (i+j*lda));
        c_00 = _mm256_add_pd(c_00, _mm256_mul_pd(a_00, b_00));
        c_00 = _mm256_add_pd(c_00, _mm256_mul_pd(a_01, b_01));
        c_00 = _mm256_add_pd(c_00, _mm256_mul_pd(a_02, b_02));
        c_00 = _mm256_add_pd(c_00, _mm256_mul_pd(a_03, b_03));
        _mm256_store_pd(C+i+j*lda, c_00);


        c_01 = _mm256_load_pd(C + (i+(j+1)*lda));
        c_01 = _mm256_add_pd(c_01, _mm256_mul_pd(a_00, b_10));
        c_01 = _mm256_add_pd(c_01, _mm256_mul_pd(a_01, b_11));
        c_01 = _mm256_add_pd(c_01, _mm256_mul_pd(a_02, b_12));
        c_01 = _mm256_add_pd(c_01, _mm256_mul_pd(a_03, b_13));
        _mm256_store_pd(C+i+(j+1)*lda, c_01);

        c_02 = _mm256_load_pd(C + (i+(j+2)*lda));
        c_02 = _mm256_add_pd(c_02, _mm256_mul_pd(a_00, b_20));
        c_02 = _mm256_add_pd(c_02, _mm256_mul_pd(a_01, b_21));
        c_02 = _mm256_add_pd(c_02, _mm256_mul_pd(a_02, b_22));
        c_02 = _mm256_add_pd(c_02, _mm256_mul_pd(a_03, b_23));
        _mm256_store_pd(C+i+(j+2)*lda, c_02);

        c_03 = _mm256_load_pd(C + (i+(j+3)*lda));
        c_03 = _mm256_add_pd(c_03, _mm256_mul_pd(a_00, b_30));
        c_03 = _mm256_add_pd(c_03, _mm256_mul_pd(a_01, b_31));
        c_03 = _mm256_add_pd(c_03, _mm256_mul_pd(a_02, b_32));
        c_03 = _mm256_add_pd(c_03, _mm256_mul_pd(a_03, b_33));
        _mm256_store_pd(C+i+(j+3)*lda, c_03);

        c_10 = _mm256_load_pd(C + (i+4+j*lda));
        c_10 = _mm256_add_pd(c_10, _mm256_mul_pd(a_10, b_00));
        c_10 = _mm256_add_pd(c_10, _mm256_mul_pd(a_11, b_01));
        c_10 = _mm256_add_pd(c_10, _mm256_mul_pd(a_12, b_02));
        c_10 = _mm256_add_pd(c_10, _mm256_mul_pd(a_13, b_03));
        _mm256_store_pd(C+i+4+j*lda, c_10);


        c_11 = _mm256_load_pd(C + (i+4+(j+1)*lda));
        c_11 = _mm256_add_pd(c_11, _mm256_mul_pd(a_10, b_10));
        c_11 = _mm256_add_pd(c_11, _mm256_mul_pd(a_11, b_11));
        c_11 = _mm256_add_pd(c_11, _mm256_mul_pd(a_12, b_12));
        c_11 = _mm256_add_pd(c_11, _mm256_mul_pd(a_13, b_13));
        _mm256_store_pd(C+i+4+(j+1)*lda, c_11);

        c_12 = _mm256_load_pd(C + (i+4+(j+2)*lda));
        c_12 = _mm256_add_pd(c_12, _mm256_mul_pd(a_10, b_20));
        c_12 = _mm256_add_pd(c_12, _mm256_mul_pd(a_11, b_21));
        c_12 = _mm256_add_pd(c_12, _mm256_mul_pd(a_12, b_22));
        c_12 = _mm256_add_pd(c_12, _mm256_mul_pd(a_13, b_23));
        _mm256_store_pd(C+i+4+(j+2)*lda, c_12);

        c_13 = _mm256_load_pd(C + (i+4+(j+3)*lda));
        c_13 = _mm256_add_pd(c_13, _mm256_mul_pd(a_10, b_30));
        c_13 = _mm256_add_pd(c_13, _mm256_mul_pd(a_11, b_31));
        c_13 = _mm256_add_pd(c_13, _mm256_mul_pd(a_12, b_32));
        c_13 = _mm256_add_pd(c_13, _mm256_mul_pd(a_13, b_33));
        _mm256_store_pd(C+i+4+(j+3)*lda, c_13);

        // c_20 = _mm256_load_pd(C + (i+8+j*lda));
        // c_20 = _mm256_add_pd(c_20, _mm256_mul_pd(a_20, b_00));
        // c_20 = _mm256_add_pd(c_20, _mm256_mul_pd(a_21, b_01));
        // c_20 = _mm256_add_pd(c_20, _mm256_mul_pd(a_22, b_02));
        // c_20 = _mm256_add_pd(c_20, _mm256_mul_pd(a_23, b_03));
        // _mm256_store_pd(C+i+8+j*lda, c_20);


        // c_21 = _mm256_load_pd(C + (i+8+(j+1)*lda));
        // c_21 = _mm256_add_pd(c_21, _mm256_mul_pd(a_20, b_10));
        // c_21 = _mm256_add_pd(c_21, _mm256_mul_pd(a_21, b_11));
        // c_21 = _mm256_add_pd(c_21, _mm256_mul_pd(a_22, b_12));
        // c_21 = _mm256_add_pd(c_21, _mm256_mul_pd(a_23, b_13));
        // _mm256_store_pd(C+i+8+(j+1)*lda, c_21);

        // c_22 = _mm256_load_pd(C + (i+8+(j+2)*lda));
        // c_22 = _mm256_add_pd(c_22, _mm256_mul_pd(a_20, b_20));
        // c_22 = _mm256_add_pd(c_22, _mm256_mul_pd(a_21, b_21));
        // c_22 = _mm256_add_pd(c_22, _mm256_mul_pd(a_22, b_22));
        // c_22 = _mm256_add_pd(c_22, _mm256_mul_pd(a_23, b_23));
        // _mm256_store_pd(C+i+8+(j+2)*lda, c_22);

        // c_23 = _mm256_load_pd(C + (i+8+(j+3)*lda));
        // c_23 = _mm256_add_pd(c_23, _mm256_mul_pd(a_20, b_30));
        // c_23 = _mm256_add_pd(c_23, _mm256_mul_pd(a_21, b_31));
        // c_23 = _mm256_add_pd(c_23, _mm256_mul_pd(a_22, b_32));
        // c_23 = _mm256_add_pd(c_23, _mm256_mul_pd(a_23, b_33));
        // _mm256_store_pd(C+i+8+(j+3)*lda, c_23);

        // c_30 = _mm256_load_pd(C + (i+12+j*lda));
        // c_30 = _mm256_add_pd(c_30, _mm256_mul_pd(a_30, b_00));
        // c_30 = _mm256_add_pd(c_30, _mm256_mul_pd(a_31, b_01));
        // c_30 = _mm256_add_pd(c_30, _mm256_mul_pd(a_32, b_02));
        // c_30 = _mm256_add_pd(c_30, _mm256_mul_pd(a_33, b_03));
        // _mm256_store_pd(C+i+12+j*lda, c_30);


        // c_31 = _mm256_load_pd(C + (i+12+ (j+1)*lda));
        // c_31 = _mm256_add_pd(c_31, _mm256_mul_pd(a_30, b_10));
        // c_31 = _mm256_add_pd(c_31, _mm256_mul_pd(a_31, b_11));
        // c_31 = _mm256_add_pd(c_31, _mm256_mul_pd(a_32, b_12));
        // c_31 = _mm256_add_pd(c_31, _mm256_mul_pd(a_33, b_13));
        // _mm256_store_pd(C+i+12+(j+1)*lda, c_31);

        // c_32 = _mm256_load_pd(C + (i+12+(j+2)*lda));
        // c_32 = _mm256_add_pd(c_32, _mm256_mul_pd(a_30, b_20));
        // c_32 = _mm256_add_pd(c_32, _mm256_mul_pd(a_31, b_21));
        // c_32 = _mm256_add_pd(c_32, _mm256_mul_pd(a_32, b_22));
        // c_32 = _mm256_add_pd(c_32, _mm256_mul_pd(a_33, b_23));
        // _mm256_store_pd(C+i+12+(j+2)*lda, c_32);

        // c_33 = _mm256_load_pd(C + (i+12+(j+3)*lda));
        // c_33 = _mm256_add_pd(c_33, _mm256_mul_pd(a_30, b_30));
        // c_33 = _mm256_add_pd(c_33, _mm256_mul_pd(a_31, b_31));
        // c_33 = _mm256_add_pd(c_33, _mm256_mul_pd(a_32, b_32));
        // c_33 = _mm256_add_pd(c_33, _mm256_mul_pd(a_33, b_33));
        // _mm256_store_pd(C+i+12+(j+3)*lda, c_33);  

      }
    }
  }
}

static inline void L1_blocking_level (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* Accumulate block dgemms into block of C */
    /* For each block-row of A */
  for (int i = 0; i < M; i += L1_BLOCK)
  {
    /* For each block-column of B */
    for (int j = 0; j < N; j += L1_BLOCK)
    {
      for (int k = 0; k < K; k += L1_BLOCK)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M1 = min (L1_BLOCK, M-i);
        int N1 = min (L1_BLOCK, N-j);
        int K1 = min (L1_BLOCK, K-k);

        /* Perform individual block dgemm */
        do_block_trial(lda, M1, N1, K1, A+i+k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}

static inline void L2_blocking_level (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* Accumulate block dgemms into block of C */
    /* For each block-row of A */
  for (int k = 0; k < K; k += L2_BLOCK)
  {
    /* For each block-column of B */
    for (int j = 0; j < N; j += L2_BLOCK)
    {
      for (int i = 0; i < M; i += L2_BLOCK)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M1 = min (L2_BLOCK, M-i);
        int N1 = min (L2_BLOCK, N-j);
        int K1 = min (L2_BLOCK, K-k);

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

  int lda_u = lda; 
  if (lda % 8){ 
      lda_u = lda + (8 - (lda % 8));
  } 

  double *Apadded = _mm_malloc(sizeof(double)*lda_u*lda_u, 64); // certain instructions (SIMD) work on contiguous segments of 8 double words
  double *Bpadded = _mm_malloc(sizeof(double)*lda_u*lda_u, 64);
  double *Cpadded = _mm_malloc(sizeof(double)*lda_u*lda_u, 64);
  memset(Cpadded, 0.0, sizeof(double)*lda_u*lda_u);

  for (int i = 0; i < lda; ++i){
       memcpy(Apadded+i*lda_u, A+i*lda, sizeof(double)*lda);
       memcpy(Bpadded+i*lda_u, B+i*lda, sizeof(double)*lda);
  }

  for (int i = lda; i < lda_u; ++i){
     memset(Apadded+i*lda_u, 0.0, sizeof(double)*lda_u);
     memset(Bpadded+i*lda_u, 0.0, sizeof(double)*lda_u);
  }

  for (int i = 0; i < lda; ++i){
       memset(Apadded+lda+i*lda_u, 0.0, sizeof(double)*(lda_u-lda));
       memset(Bpadded+lda+i*lda_u, 0.0, sizeof(double)*(lda_u-lda));
  }


  /* For each block-column of B */
  for (int j = 0; j < lda_u; j += L3_BLOCK)
  {
    /* Accumulate block dgemms into block of C */
    for (int i = 0; i < lda_u; i += L3_BLOCK)
    {
      /* For each block-row of A */ 
      for (int k = 0; k < lda_u; k += L3_BLOCK)
      {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M = min (L3_BLOCK, lda_u-i);
          int N = min (L3_BLOCK, lda_u-j);
          int K = min (L3_BLOCK, lda_u-k);
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


