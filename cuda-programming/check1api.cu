#include "error.cuh"
#include<math.h>
#include<stdio.h>

#ifdef USE_DP
typedef double real;
    const real EPSILON = 1.0e-15;
#else
typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

const int NUM_REPEATS = 10;
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;

void __global__ add(const real *x, const real *y, real *z, const int N);

void check(const real *z, const int N);

int main(void) {
    int N = 100000000;
    real * h_x = new real[N];
    real * h_y = new real[N];
    real * h_z = new real[N];
    int M = N * sizeof(real);

    for (int i = 0; i < N; i++) {
        h_x[i] = a;
        h_y[i] = b;
    }

    real * d_x, *d_y, *d_z;
    CHECK(cudaMalloc((void **) &d_x, M));
    CHECK(cudaMalloc((void **) &d_y, M));
    CHECK(cudaMalloc((void **) &d_z, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    int block_size = 128;
    int grid_size = (N - 1) / block_size + 1;


    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0) {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}

void __global__ add(const real *x, const real *y, real *z, const int N) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) {
        z[n] = x[n] + y[n];
    }
}

void check(const real *z, const int N) {
    bool has_error = false;
    for (int i = 0; i < N; i++) {
        if (fabs(z[i] - c) > EPSILON) {
            has_error = true;
        }
    }
    if (has_error) {
        printf("has error");
    } else {
        printf("well done");
    }

}