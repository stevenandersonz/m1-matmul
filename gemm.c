#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h> 
#define TOLERANCE 1e-3 
#define N 1024
#define BLOCK_SIZE 64 

float A[N][N];
float B[N][N];
float C[N][N];
float D[N][N];
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
int main (){
    //generate arrays and compute numpy time
    system("python3 gemm.py");
    // init arrays
    load_m("A.txt", A);
    load_m("B.txt", B);
    uint64_t start = nanos();

    // compute
    for (int by=0; by<N; by+=BLOCK_SIZE){
        for (int bx=0; bx<N; bx+=BLOCK_SIZE){
            for (int k=0; k<N; k+=BLOCK_SIZE){
                for(int y = by; y < by + BLOCK_SIZE; y++) {
                    for(int x = bx; x < bx + BLOCK_SIZE; x++) {
                        for(int k1 = k; k1 < k + BLOCK_SIZE; k1++) {
                            C[y][x] += A[y][k1] * B[k1][x];
                        }
                    }
                }
            }
        }
    }
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