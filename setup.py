from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="live_eplb_cuda",
    ext_modules=[
        CUDAExtension(
            name="live_eplb_cuda",
            sources=["csrc/p2p_swapper.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
