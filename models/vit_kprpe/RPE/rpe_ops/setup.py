import os

from setuptools import setup, Extension
import torch
from torch.utils import cpp_extension

ext_t = cpp_extension.CppExtension
ext_fnames = ['rpe_index.cpp']
define_macros = []
extra_compile_args = dict(cxx=['-fopenmp', '-O3'],
                          nvcc=['-O3'])

force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
has_cuda_toolkit = cpp_extension.CUDA_HOME is not None

if force_cuda or has_cuda_toolkit:
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = os.getenv(
            "RPE_TORCH_CUDA_ARCH_LIST",
            "8.0;8.6;8.9;9.0;12.0+PTX",
        )
    ext_t = cpp_extension.CUDAExtension
    ext_fnames.append('rpe_index_cuda.cu')
    define_macros.append(('WITH_CUDA', None))

setup(name='rpe_index',
      version="1.2.0",
      ext_modules=[ext_t(
                   'rpe_index_cpp',
                   ext_fnames,
                   define_macros=define_macros,
                   extra_compile_args=extra_compile_args,
                   )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
