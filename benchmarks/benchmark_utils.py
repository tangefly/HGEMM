import matplotlib.pyplot as plt
import os
import time
import torch

def time_it(fn, iters, sync=None):
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        sync()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def run_op_benchmark(
    name,
    fn,
    flops,
    iters=30,
    warmup=10,
    sync=None,
):
    # warmup
    for _ in range(warmup):
        fn()
    if sync:
        sync()

    t = time_it(fn, iters, sync)
    return {
        "name": name,
        "time": t,
        "tflops": flops / t / 1e12,
    }


def benchmark_square_gemm(
    size,
    gpu_ops,
    cpu_ops,
    iters=30,
    warmup=10,
):
    M = N = K = size
    flops = 2 * M * N * K

    # ---------------- GPU tensors ----------------
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    results = []

    for name, op in gpu_ops.items():
        fn = lambda op=op: op(A, B)
        res = run_op_benchmark(
            name=name,
            fn=fn,
            flops=flops,
            iters=iters,
            warmup=warmup,
            sync=torch.cuda.synchronize,
        )
        results.append(res)

    # ---------------- CPU tensors ----------------
    A_cpu = torch.randn(M, K, device="cpu", dtype=torch.float32)
    B_cpu = torch.randn(K, N, device="cpu", dtype=torch.float32)

    for name, op in cpu_ops.items():
        fn = lambda op=op: op(A_cpu, B_cpu)
        res = run_op_benchmark(
            name=name,
            fn=fn,
            flops=flops,
            iters=iters,
            warmup=warmup,
            sync=None,
        )
        results.append(res)

    return results

def save_plot_result(
    sizes: list,
    perf_data: dict, 
    out_dir: str, 
    out_name: str,
    title: str
):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    plt.figure(figsize=(8, 6))

    for name, tflops in perf_data.items():
        plt.plot(sizes, tflops, marker="o", label=name)

    plt.xlabel("Matrix size (M = N = K)")
    plt.ylabel("TFLOPS")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()



    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\nSaved figure to: {out_path}")


