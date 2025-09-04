import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# load the CUDA kernel as python module
lib = load(
    name="relu_lib",
    sources=['relu.cu'],
    extra_cuda_cflags=[
        '-O3',
        '-U__CUDA_NOHALF_OPERATIORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '--use_fast_math',
    ],
    extra_cflags=['-std=c++17'],
)

def run_benchmark(
    perf_func: callable,
    x: torch.Tensor, 
    tag:str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)
    torch.cuda.sychronize()

    # iter
    start = time.time()
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            _ = perf_func(x)   
    torch.cuda.sychronize()
    end = time.time()

    total_time = (end - start) * 1000
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().deatch().cpu().numpy().tolist()[:2]
    out_val = [round(v,8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    
    return out, mean_time

x_dim = [1024, 2048, 4096]
y_dim = [1024, 2048, 4096]

xy_dim = [(x, y) for x in x_dim for y in y_dim]

for x_dim, y_dim in (xy_dim):
    print("-" * 85)
    print(" " * 40 + f"x_dim={x_dim}, y_dim={y_dim}")
    input_x = torch.randn((x_dim, y_dim)).cuda().float().contiguous()
    output_y = torch.zeros_like(input_x).cuda().float().contiguous()

    lib.relu_f32(input_x, output_y)
    th_out = torch.relu(input_x)

    print(output_y[:5])
    print(th_out[:5])

    print('...')