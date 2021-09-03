#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudautils.h"
#include "cudaImage.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub, cudaStream_t stream);
void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp, cudaStream_t stream);
double ScaleDown(CudaImage &res, CudaImage &src, float variance, cudaStream_t stream);
double ScaleUp(CudaImage &res, CudaImage &src, cudaStream_t stream);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave, cudaStream_t stream);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave, cudaStream_t stream);
double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave, cudaStream_t stream);
double RescalePositions(SiftData &siftData, float scale, cudaStream_t stream);
double LowPass(CudaImage &res, CudaImage &src, float scale, cudaStream_t stream);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave, cudaStream_t stream);
double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave, cudaStream_t stream);

#endif
