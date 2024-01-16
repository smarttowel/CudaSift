project('cudasift', 'cpp', 'cuda',
    version: '1.0.0',
    default_options: ['cpp_std=c++14'])

cuda_versions = ['12.2', '11.7', '11.4', '10.2']
cuda_dep = []
foreach version: cuda_versions
    cuda_dep = [dependency('cudart-' + version, required: false),
                dependency('cuda-' + version, required: false)]
    if(cuda_dep[0].found() and cuda_dep[1].found())
        break
    endif
endforeach
assert(cuda_dep[0].found() and cuda_dep[1].found(), 'cuda not found')

cudasift_lib = static_library('cudasift',
                              'cudaImage.cu',
                              'cudaSiftD.cu',
                              'cudaSiftH.cu',
                              'matching.cu',
                               cuda_args: ['--Wno-deprecated-declarations'],
                               dependencies: [cuda_dep])
cudasift_dep = declare_dependency(link_with: cudasift_lib,
                                  include_directories: include_directories('.'))
