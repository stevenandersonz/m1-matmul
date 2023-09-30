#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h> 
#include <string.h> 
#include <arm_neon.h>

#define TOLERANCE 1e-3 
#define N 1024
#define BLOCK_X 16
#define BLOCK_K 8 
#define BLOCK_Y 8 
#define BLOCK 8 

float A[N*N] __attribute__ ((aligned(16)));
float B[N*N] __attribute__ ((aligned(16)));
float C[N*N] __attribute__ ((aligned(16)));
float D[N*N] __attribute__ ((aligned(16)));
uint64_t nanos(){
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return start.tv_nsec + start.tv_sec * 1000000000;
}
void load_m (const char *nm, float* const M){
    FILE *fp = fopen(nm, "r");
    if (fp == NULL) {
        printf("Error opening file.\n");
        return;
    }
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fscanf(fp, "%f", &M[i*N+j]);
        }
    }

    fclose(fp);
}
void matmul(float* const A, float* const B, float* const C){
    for(int by = 0; by < N; by+=BLOCK_Y) {
        for(int bx = 0; bx < N; bx+=BLOCK_X) {
            for(int bk = 0; bk < N; bk+=BLOCK_K) {
                for (int y=by; y<by+BLOCK_Y; y++){
                    for (int x=bx; x<bx+BLOCK_X; x+=4){
                        float32x4_t sum_vec = vld1q_f32(&C[y * N + x]);
                        for (int k=bk; k<bk+BLOCK_K; k++){
                            float32x4_t a_vec = vdupq_n_f32(A[y * N + k]);
                            float32x4_t b_vec = vld1q_f32(&B[k*N+x]);
                            sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
                        }
                        vst1q_f32(&C[y * N + x], sum_vec);
                    }
                }
            }
        }
    }
}
// for (int y = sy; y < ey; y+=BLOCK_Y) {
//     for (int x = 0; x < N; x+=BLOCK*BLOCK_X) {

//       __m256 acc[BLOCK_Y][BLOCK_X] = {};
//       for (int k = 0; k < N; k++) {
//         for (int iy = 0; iy < BLOCK_Y; iy++) {
//           __m256 ta = _mm256_broadcast_ss(&A[(y+iy)*N + k]);
//           for (int ix = 0; ix < BLOCK_X; ix++) {
//             acc[iy][ix] = _mm256_fmadd_ps(ta, Bfm[((x+ix*BLOCK)*N + k*8)/8], acc[iy][ix]);
//           }
//         }
//       }

//       for (int iy = 0; iy < BLOCK_Y; iy++) {
//         for (int ix = 0; ix < BLOCK_X; ix++) {
//           Cm[((y+iy)*N + x + ix * BLOCK)/8] = acc[iy][ix];
//         }
//       }
//     }
//   }
int main (){
    //generate arrays and compute numpy time
    system("python3 gemm.py");
    // init arrays
    load_m("A.txt", A);
    load_m("B.txt", B);
        uint64_t start = nanos();
        matmul(A,B,C);
    //  matmul_neon(&A[0][0], &B[0][0], &C[0][0]);
        uint64_t end = nanos();
        double tflop = (2.0*N*N*N) * 1e-9;
        double s = (end - start) * 1e-9;
        printf("C's %f GFLOP/S \n", tflop/s);
   // verify
    load_m("C.txt", D);
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            if(fabs(D[i*N+j]-C[i*N+j])>TOLERANCE){
                printf("\n %f != %f at position %d,%d \n", D[i*N+j], C[i*N+j], i, j);
            }
        }
    }

}