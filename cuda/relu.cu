#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define MAX_BLOCK_THREAD 1024

// x:N y:N
// y = max(0, x)
// block(256) grid(N / 256)


__global__ void relu_f32_kernel(float *x, float *y, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        y[idx] = fmaxf(0.0f, x[idx]);

}


void relu_f32(torch::Tensor x, torch::Tensor y){
    // check type
    if (x.options().dtype() != torch::kFloat32){
        std::cout << "Tensor info:" << x.options() << std::endl;
        throw std::runtime_error("value must be torch::kFloat32\n");
    }

    if (y.options().dtype() != torch::kFloat32){
        std::cout << "Tensor info:" << y.options() << std::endl;
        throw std::runtime_error("value must be torch::kFloat32\n");
    }

    const int ndim = x.dim();
    if (ndim == 2){
        int x_dim = x.size(0);
        int y_dim = x.size(1);
        int N = x_dim * y_dim;

        if (x_dim <= MAX_BLOCK_THREAD){
            dim3 block(x_dim);
            dim3 grid(y_dim);
            relu_f32_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()), reinterpret_cast<float *>(y.data_ptr()), N);
        }else{
            dim3 block(256);
            dim3 grid((N + 256 -1) / 256);
            relu_f32_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()), reinterpret_cast<float *>(y.data_ptr()), N);
        }
    }else{
        int N = 1;
        for(int i=0; i< ndim; i++){N *= x.size(i);}
        dim3 block(256);
        dim3 grid((N + 256 - 1) / 256);
        relu_f32_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()), reinterpret_cast<float *>(y.data_ptr()), N);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("relu_f32", &relu_f32, "This is a relu_f32_kernel interface...");
}


int main(){
    // prepare host mem

    // prepare device mem

    // cpu mem -> gpu mem

    // prepare grid and block

    // launch kernel

    // relu_f32_kernel<<< grid, block >>>(x, y, N);

    // gpu mem -> cpu mem

    // check result

    // free mem
    return 0;
}