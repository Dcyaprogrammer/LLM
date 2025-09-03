#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

void check_result(float *host_ref, float *dev_ref, const int N){
    double var = 1e-8;
    int flag = 1;
    for(int i=0; i < N; i++){
        if(abs(host_ref[i] - dev_ref[i]) > var){
            flag = 0;
            break;
        }
    }

    if(flag==0)
        printf("Wrong Not Match");
    else
        printf("Match\n");
}

// compute on cpu
void tensor_add_cpu(float *a, float *b, float *res, const int N){
    for(int i=0; i < N; i++){
        res[i] = a[i] + b[i];
    }
}

// compute on gpu
__global__ void tensor_add_gpu(float *a, float *b, float *res, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    res[i] = a[i] + b[i]; 
}

void initialize_data(float *a, const int N){
    for (int i = 0; i < N; i++){
        a[i] = i;
    }
}

int main(){

    cudaSetDevice(0);

    int N = 1024;
    size_t nBytes = N * sizeof(float);

    float *h_a, *h_b, *h_res;
    float *d_a, *d_b, *d_res;
    float *ground_truth;

    h_a = (float *)malloc(nBytes);
    h_b = (float *)malloc(nBytes);
    h_res = (float *)malloc(nBytes);
    ground_truth = (float *)malloc(nBytes);

    initialize_data(h_a, N);
    initialize_data(h_b, N);

    cudaMalloc((float**)(&d_a), nBytes);
    cudaMalloc((float**)(&d_b), nBytes);
    cudaMalloc((float**)(&d_res), nBytes);

    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(N / 32);

    tensor_add_cpu(h_a, h_b, ground_truth, N);
    
    tensor_add_gpu<<< grid, block >>>(d_a, d_b, d_res, N);
    printf("Begin computing on gpu...\n");

    cudaMemcpy(h_res, d_res, nBytes, cudaMemcpyDeviceToHost);

    check_result(ground_truth, h_res, N);

    // free gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    // free cpu memory
    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_res);
    cudaFree(ground_truth);


    return 0;

}