#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h> 
#include <string.h> 
#include <arm_neon.h>

#define TOLERANCE 1e-3 
#define N 1024
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_K 8 
#define BLOCK_SIZE_Y 8 

float A[N][N] __attribute__ ((aligned(32)));
float B[N][N] __attribute__ ((aligned(32)));
float C[N][N]  __attribute__ ((aligned(32)));
float D[N][N] __attribute__ ((aligned(32)));
uint64_t nanos(){
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return start.tv_nsec + start.tv_sec * 1000000000;
}
void load_m (const char *nm, float M[N][N]){
    FILE *fp = fopen(nm, "r");
    if (fp == NULL) {
        printf("Error opening file.\n");
        return;
    }
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fscanf(fp, "%f", &M[i][j]);
        }
    }

    fclose(fp);
}
void simple_matmul_neon(float *A, float *B, float *C) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f); // Initialize to zero

            for(int k = 0; k < N; k += 4) {
                // Load vectors from A and B
                float32x4_t a_vec = vld1q_f32(&A[i*N + k]);
                float b_values[4] = { B[k * N + j], B[(k+1) * N + j], B[(k+2) * N + j], B[(k+3) * N + j] };
                float32x4_t b_vec = vld1q_f32(b_values);
                // float32x4_t b_vec = vld1q_f32(&B[k*N + j]);

                // Multiply and accumulate
                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
            }

            // Add up the elements in sum_vec and store in C
            float sum_array[4];
            vst1q_f32(sum_array, sum_vec);
            C[i*N + j] = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        }
    }
}
void matmul_neon(float *A, float *B, float *C) {
    for(int by = 0; by < N; by+=BLOCK_SIZE_Y) {
        for(int bx = 0; bx < N; bx+=BLOCK_SIZE_X) {
            float32x4_t acc[BLOCK_SIZE_Y][BLOCK_SIZE_X] = {};
            for(int bk = 0; bk < N; bk++) {
                for(int y = by; y < by; y++) {
                    for(int x = bx; x < bx; x++) {
                        //float32x4_t ta = vld1q_dup_f32(&A[(y+iy)*N + k]);
                        float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
                         for(int k = bk; k < bk+BLOCK_SIZE_Y; k+=4) {
                            float32x4_t a_vec = vld1q_f32(&A[y*N+k]);
                            //float32x4_t a_vec = vld1q_dup_f32(&A[by+y*N+k]);
                            float32x4_t b_vec0 = {B[k * N + x], B[(k + 1) * N + x], B[(k + 2) * N + x], B[(k + 3) * N + x]};
                            sum_vec0 = vmlaq_f32(sum_vec0, a_vec, b_vec0);
                        }
                        C[y*N+x] += vaddvq_f32(sum_vec0);
                    }
                }
            } 
        }
    }
}
void matmul(float* const A, float* const B, float* const C){
    for (int y=0; y<N; y++){
        for (int k=0; k<N; k++){
            float32x4_t a_vec = vdupq_n_f32( A[y * N + k]);
            for (int x=0; x<N; x+=4){
                float32x4_t sum_vec = vld1q_f32(&C[y * N + x]);
                float32x4_t b_vec = vld1q_f32(&B[k*N+x]);
                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
                vst1q_f32(&C[y * N + x], sum_vec);
            }
        }
    }
}
int main (){
    //generate arrays and compute numpy time
    system("python3 gemm.py");
    // init arrays
    load_m("A.txt", A);
    load_m("B.txt", B);
        uint64_t start = nanos();
        matmul(&A[0][0],&B[0][0],&C[0][0]);
    //  matmul_neon(&A[0][0], &B[0][0], &C[0][0]);
        uint64_t end = nanos();
        double tflop = (2.0*N*N*N) * 1e-9;
        double s = (end - start) * 1e-9;
        printf("C's %f GFLOP/S \n", tflop/s);
   // verify
    load_m("C.txt", D);
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            if(fabs(D[i][j]-C[i][j])>TOLERANCE){
                printf("\n %f != %f at position %d,%d \n", D[i][j], C[i][j], i, j);
            }
        }
    }

}