project('cudasift', 'cpp',
    version: '1.0.0',
    default_options: ['cpp_std=c++14'])

cuda_dep = dependency('cudart-10.2')

cudasift_lib = static_library('cudasift',
                              'cudaImage.cu',
                              'cudaSiftD.cu',
                              'cudaSiftH.cu',
                              'matching.cu',
                               dependencies: [cuda_dep],
                               override_options: ['cpp_std=c++14'])
cudasift_dep = declare_dependency(link_with: cudasift_lib,
                                  include_directories: include_directories('.'))
