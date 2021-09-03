//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//  

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "cudautils.h"

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD.h"
#include "cudaSiftH.h"

#include "cudaSiftD.cu"

void InitCuda(int devNum)
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (!nDevices) {
    std::cerr << "No CUDA devices available" << std::endl;
    return;
  }
  devNum = std::min(nDevices-1, devNum);
  deviceInit(devNum);  
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devNum);
  printf("Device Number: %d\n", devNum);
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1000);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n",
	 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp)
{
  const int nd = NUM_SCALES + 3;
  int w = width*(scaleUp ? 2 : 1); 
  int h = height*(scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int size = h*p;                 // image sizes
  int sizeTmp = nd*h*p;           // laplace buffer sizes
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
    sizeTmp += nd*h*p; 
  }
  float *memoryTmp = NULL; 
  size_t pitch;
  size += sizeTmp;
  safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
  return memoryTmp;
}

void FreeSiftTempMemory(float *memoryTmp)
{
  if (memoryTmp)
    safeCall(cudaFree(memoryTmp));
}

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory, cudaStream_t stream) 
{
  unsigned int *d_PointCounterAddr;
  safeCall(cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter));
  safeCall(cudaMemsetAsync(d_PointCounterAddr, 0, (8*2+1)*sizeof(int), stream));
  safeCall(cudaMemcpyToSymbolAsync(d_MaxNumPoints, &siftData.maxPts, sizeof(int), 0, cudaMemcpyHostToDevice, stream));

  const int nd = NUM_SCALES + 3;
  int w = img.width*(scaleUp ? 2 : 1);
  int h = img.height*(scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int width = w, height = h;
  int size = h*p;                 // image sizes
  int sizeTmp = nd*h*p;           // laplace buffer sizes
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
    sizeTmp += nd*h*p; 
  }
  float *memoryTmp = tempMemory; 
  size += sizeTmp;
  if (!tempMemory) {
    size_t pitch;
    safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
  }
  float *memorySub = memoryTmp + sizeTmp;

  CudaImage lowImg;
  lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);
  if (!scaleUp) {
    float kernel[8*12*16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
    safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8*12*16*sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    LowPass(lowImg, img, max(initBlur, 0.001f), stream);
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128), stream);
    safeCall(cudaMemcpyAsync(&siftData.numPts, &d_PointCounterAddr[2*numOctaves], sizeof(int), cudaMemcpyDeviceToHost, stream));
    safeCall(cudaStreamSynchronize(stream));
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
  } else {
    CudaImage upImg;
    upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
    ScaleUp(upImg, img, stream);
    LowPass(lowImg, upImg, max(initBlur, 0.001f), stream);
    float kernel[8*12*16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
    safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8*12*16*sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale*2.0f, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128), stream);
    safeCall(cudaMemcpyAsync(&siftData.numPts, &d_PointCounterAddr[2*numOctaves], sizeof(int), cudaMemcpyDeviceToHost, stream)); 
    safeCall(cudaStreamSynchronize(stream));
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
    RescalePositions(siftData, 0.5f, stream);
  } 
  
  if (!tempMemory)
    safeCall(cudaFree(memoryTmp));
#ifdef MANAGEDMEM
  safeCall(cudaStreamSynchronize(stream));
#else
  if (siftData.h_data)
    safeCall(cudaMemcpyAsync(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*siftData.numPts, cudaMemcpyDeviceToHost, stream));
#endif
}

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub, cudaStream_t stream) 
{
  int w = img.width;
  int h = img.height;
  if (numOctaves>1) {
    CudaImage subImg;
    int p = iAlignUp(w/2, 128);
    subImg.Allocate(w/2, h/2, p, false, memorySub); 
    ScaleDown(subImg, img, 0.5f, stream);
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, numOctaves-1, totInitBlur, thresh, lowestScale, subsampling*2.0f, memoryTmp, memorySub + (h/2)*p, stream);
  }
  ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, subsampling, memoryTmp, stream);
  return 0;
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp, cudaStream_t stream)
{
  const int nd = NUM_SCALES + 3;
  CudaImage diffImg[nd];
  int w = img.width; 
  int h = img.height;
  int p = iAlignUp(w, 128);
  for (int i=0;i<nd-1;i++) 
    diffImg[i].Allocate(w, h, p, false, memoryTmp + i*p*h); 

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = img.d_data;
  resDesc.res.pitch2D.width = img.width;
  resDesc.res.pitch2D.height = img.height;
  resDesc.res.pitch2D.pitchInBytes = img.pitch*sizeof(float);  
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  float baseBlur = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  LaplaceMulti(texObj, img, diffImg, octave, stream); 
  FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f/NUM_SCALES, lowestScale/subsampling, subsampling, octave, stream);
  ComputeOrientations(texObj, img, siftData, octave, stream); 
  ExtractSiftDescriptors(texObj, siftData, subsampling, octave, stream); 
  //OrientAndExtract(texObj, siftData, subsampling, octave); 
  
  safeCall(cudaDestroyTextureObject(texObj));
}

void InitSiftData(SiftData &data, int num, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = num;
  int sz = sizeof(SiftPoint)*num;
#ifdef MANAGEDMEM
  safeCall(cudaMallocManaged((void **)&data.m_data, sz));
#else
  data.h_data = NULL;
  if (host)
    data.h_data = (SiftPoint *)malloc(sz);
  data.d_data = NULL;
  if (dev)
    safeCall(cudaMalloc((void **)&data.d_data, sz));
#endif
}

void FreeSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
  safeCall(cudaFree(data.m_data));
#else
  if (data.d_data!=NULL)
    safeCall(cudaFree(data.d_data));
  data.d_data = NULL;
  if (data.h_data!=NULL)
    free(data.h_data);
#endif
  data.numPts = 0;
  data.maxPts = 0;
}

void PrintSiftData(SiftData &data, cudaStream_t stream)
{
#ifdef MANAGEDMEM
  SiftPoint *h_data = data.m_data;
#else
  SiftPoint *h_data = data.h_data;
  if (data.h_data==NULL) {
    h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*data.maxPts);
    safeCall(cudaMemcpyAsync(h_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToHost, stream));
    safeCall(cudaStreamSynchronize(stream));
    data.h_data = h_data;
  }
#endif
  for (int i=0;i<data.numPts;i++) {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score        = %.2f\n", h_data[i].score);
    float *siftData = (float*)&h_data[i].data;
    for (int j=0;j<8;j++) {
      if (j==0) 
	printf("data = ");
      else 
	printf("       ");
      for (int k=0;k<16;k++)
	if (siftData[j+8*k]<0.05)
	  printf(" .   ");
	else
	  printf("%.2f ", siftData[j+8*k]);
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data.numPts);
  printf("Number of allocated points: %d\n", data.maxPts);
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double ScaleDown(CudaImage &res, CudaImage &src, float variance, cudaStream_t stream)
{
  static float oldVariance = -1.0f;
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
  if (oldVariance!=variance) {
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j=0;j<5;j++) {
      h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);      
      kernelSum += h_Kernel[j];
    }
    for (int j=0;j<5;j++)
      h_Kernel[j] /= kernelSum;  
    safeCall(cudaMemcpyToSymbolAsync(d_ScaleDownKernel, h_Kernel, 5*sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    oldVariance = variance;
  }
#if 0
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4, SCALEDOWN_H + 4);
  ScaleDownDenseShift<<<blocks, threads, 0, stream>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#else
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  ScaleDown<<<blocks, threads, 0, stream>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#endif
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double ScaleUp(CudaImage &res, CudaImage &src, cudaStream_t stream)
{
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleUp: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
  dim3 threads(SCALEUP_W/2, SCALEUP_H/2);
  ScaleUp<<<blocks, threads, 0, stream>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch); 
  checkMsg("ScaleUp() execution failed\n");
  return 0.0;
}   

double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave, cudaStream_t stream)
{
  dim3 blocks(512); 
#ifdef MANAGEDMEM
  ComputeOrientationsCONST<<<blocks, threads, 0, stream>>>(texObj, siftData.m_data, octave);
#else
#if 1
  dim3 threads(11*11);
  ComputeOrientationsCONST<<<blocks, threads, 0, stream>>>(texObj, siftData.d_data, octave);
#else
  dim3 threads(256); 
  ComputeOrientationsCONSTNew<<<blocks, threads, 0, stream>>>(src.d_data, src.width, src.pitch, src.height, siftData.d_data, octave);
#endif
#endif
  checkMsg("ComputeOrientations() execution failed\n");
  return 0.0;
}

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave, cudaStream_t stream)
{
  dim3 blocks(512); 
  dim3 threads(16, 8);
#ifdef MANAGEDMEM
  ExtractSiftDescriptorsCONST<<<blocks, threads, 0, stream>>>(texObj, siftData.m_data, subsampling, octave);
#else
  ExtractSiftDescriptorsCONSTNew<<<blocks, threads, 0, stream>>>(texObj, siftData.d_data, subsampling, octave);
#endif
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return 0.0; 
}

double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave, cudaStream_t stream)
{
  dim3 blocks(256); 
  dim3 threads(128);
#ifdef MANAGEDMEM
  OrientAndExtractCONST<<<blocks, threads, 0, stream>>>(texObj, siftData.m_data, subsampling, octave);
#else
  OrientAndExtractCONST<<<blocks, threads, 0, stream>>>(texObj, siftData.d_data, subsampling, octave);
#endif
  checkMsg("OrientAndExtract() execution failed\n");
  return 0.0;
}

double RescalePositions(SiftData &siftData, float scale, cudaStream_t stream)
{
  dim3 blocks(iDivUp(siftData.numPts, 64));
  dim3 threads(64);
  RescalePositions<<<blocks, threads, 0, stream>>>(siftData.d_data, siftData.numPts, scale);
  checkMsg("RescapePositions() execution failed\n");
  return 0.0; 
}

double LowPass(CudaImage &res, CudaImage &src, float scale, cudaStream_t stream)
{
  float kernel[2*LOWPASS_R+1];
  static float oldScale = -1.0f;
  if (scale!=oldScale) {
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*scale*scale);
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) {
      kernel[j+LOWPASS_R] = (float)expf(-(double)j*j*ivar2);
      kernelSum += kernel[j+LOWPASS_R]; 
    }
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) 
      kernel[j+LOWPASS_R] /= kernelSum;  
    safeCall(cudaMemcpyToSymbolAsync(d_LowPassKernel, kernel, (2*LOWPASS_R+1)*sizeof(float), 0, cudaMemcpyHostToDevice, stream));
    oldScale = scale;
  }  
  int width = res.width;
  int pitch = res.pitch;
  int height = res.height;
  dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
#if 1
  dim3 threads(LOWPASS_W+2*LOWPASS_R, 4); 
  LowPassBlock<<<blocks, threads, 0, stream>>>(src.d_data, res.d_data, width, pitch, height);
#else
  dim3 threads(LOWPASS_W+2*LOWPASS_R, LOWPASS_H);
  LowPass<<<blocks, threads, 0, stream>>>(src.d_data, res.d_data, width, pitch, height);
#endif
  checkMsg("LowPass() execution failed\n");
  return 0.0; 
}

//==================== Multi-scale functions ===================//

void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel)
{
  if (numOctaves>1) {
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    PrepareLaplaceKernels(numOctaves-1, totInitBlur, kernel);
  }
  float scale = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  for (int i=0;i<NUM_SCALES+3;i++) {
    float kernelSum = 0.0f;
    float var = scale*scale - initBlur*initBlur;
    for (int j=0;j<=LAPLACE_R;j++) {
      kernel[numOctaves*12*16 + 16*i + j] = (float)expf(-(double)j*j/2.0/var);
      kernelSum += (j==0 ? 1 : 2)*kernel[numOctaves*12*16 + 16*i + j]; 
    }
    for (int j=0;j<=LAPLACE_R;j++)
      kernel[numOctaves*12*16 + 16*i + j] /= kernelSum;
    scale *= diffScale;
  }
}
 
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave, cudaStream_t stream) 
{
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
#if 1
  dim3 threads(LAPLACE_W+2*LAPLACE_R);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMem<<<blocks, threads, 0, stream>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), iDivUp(height, LAPLACE_H));
  LaplaceMultiMemTest<<<blocks, threads, 0, stream>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMemOld<<<blocks, threads, 0, stream>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiTex<<<blocks, threads, 0, stream>>>(texObj, results[0].d_data, width, pitch, height, octave);
#endif
  checkMsg("LaplaceMulti() execution failed\n");
  return 0.0; 
}

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave, cudaStream_t stream)
{
  if (sources->d_data==NULL) {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
#if 0
  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2, MINMAX_H);
  FindPointsMultiTest<<<blocks, threads, 0, stream>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave); 
#endif
#if 1
  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2); 
#ifdef MANAGEDMEM
  FindPointsMulti<<<blocks, threads, 0, stream>>>(sources->d_data, siftData.m_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave); 
#else
  FindPointsMultiNew<<<blocks, threads, 0, stream>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
#endif
  checkMsg("FindPointsMulti() execution failed\n");
  return 0.0;
}

