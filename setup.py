from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="hgemm_ext",
    ext_modules=[
        CUDAExtension(
            name="hgemm_ext._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/kernels/hgemm.cu",
                "csrc/kernels/naive_hgemm.cu",
                "csrc/kernels/hierarchical_tiling_hgemm.cu",
                "csrc/kernels/thread_tile_hgemm.cu",
                "csrc/kernels/warp_hgemm.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["hgemm_ext"],
)
