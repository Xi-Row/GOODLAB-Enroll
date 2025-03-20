from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_layernorm',
    ext_modules=[
        CUDAExtension(
            name='custom_layernorm',
            sources=['layer_norm_cuda.cu'],
            extra_compile_args={'cxx': [], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)