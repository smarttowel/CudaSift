project('cudasift', 'cpp', 'cuda',
    version: '1.0.0',
    default_options: ['cpp_std=c++14'])

cuda_dep = dependency('cudart-10.2')

cudasift_lib = static_library('cudasift',
                              'cudaImage.cu',
                              'cudaSiftD.cu',
                              'cudaSiftH.cu',
                              'matching.cu',
                               cuda_args: ['--Wno-deprecated-declarations'],
                               dependencies: [cuda_dep])
cudasift_dep = declare_dependency(link_with: cudasift_lib,
                                  include_directories: include_directories('.'))
