from benchmark_utils import benchmark_square_gemm, save_plot_result
from collections import defaultdict
import hgemm_ext
import os
import torch

gpu_ops = {
    "thread_tile_hgemm": hgemm_ext.thread_tile_hgemm,
    "cuBLAS": torch.matmul,
}

cpu_ops = {
    "CPU_matmul": torch.matmul,
}

sizes = [256, 512, 1024, 1536, 2048, 3072, 4096]

perf_data = defaultdict(list)
for s in sizes:
    results = benchmark_square_gemm(s, gpu_ops, cpu_ops)
    for r in results:
        perf_data[r["name"]].append(r["tflops"])

out_dir = "./results"
out_name = "thread_tile_hgemm_benchmark.png"
title = "Thread Tile HGEMM Performance Comparison"

save_plot_result(sizes, perf_data, out_dir, out_name, title)