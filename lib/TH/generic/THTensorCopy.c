#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else
#define TH_OMP_THRESHOLD 10
void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  int dimI = src->nDimension;
  int dimO = tensor->nDimension;

  int strideConI = 1;
  int strideConO = 1;
  int iter = 0;
  for(iter = 0; iter < dimI; iter++)
  {
    if(0 == src->stride[iter]) 
      strideConI = 0;
  }
  for(iter = 0; iter < dimO; iter++)
  {
    if(0 == tensor->stride[iter]) 
      strideConO = 0;
  }

  if (THTensor_(isContiguous)(src) && THTensor_(isContiguous)(tensor) && (1 == src->stride[src->nDimension-1]) && (1 == tensor->stride[tensor->nDimension-1])&& THTensor_(nElement)(src) == THTensor_(nElement)(tensor))
  {
    real *tp = THTensor_(data)(src);
    real *rp = THTensor_(data)(tensor);
    long long sz = THTensor_(nElement)(src);
    long long j;
    #pragma omp parallel for if(sz > TH_OMP_THRESHOLD) private(j)
    #pragma ivdep
    for (j=0; j<sz; j++)
    {
      rp[j] = tp[j];
    }
  }
  else if((dimI==dimO) && (dimI==3) && (strideConI != 0) && (strideConO != 0) && THTensor_(isContiguous)(tensor)  && (1 == src->stride[2]) && (1 == tensor->stride[2]) && THTensor_(nElement)(src) == THTensor_(nElement)(tensor))
  {
    real *tp = THTensor_(data)(src);
    real *rp = THTensor_(data)(tensor);
    long long sz = THTensor_(nElement)(tensor);
    long long srcBasicIndex = 0;
    
    long i = 0;
    long long k = 0;
    long jj = 0;
    long long kk = 0;
    real *rpLocal = NULL;
    real *tpLocal = NULL;
    #pragma omp parallel for private(k)      
    for(k=0; k < sz; k+=tensor->stride[1])
    { 
      kk = k/tensor->stride[0];
      jj = (k%tensor->stride[0])/tensor->stride[1];
      srcBasicIndex = kk*src->stride[0] + jj*src->stride[1];
      rpLocal = rp+k;
      tpLocal = tp+srcBasicIndex;
      #pragma ivdep
      for(i=0; i < tensor->stride[1]; i++)
      {
        rpLocal[i] = tpLocal[i];
      }
    }
  }
  else if((dimI==dimO) && (dimI == 4) && (strideConI != 0) && (strideConO != 0) &&  THTensor_(isContiguous)(tensor) && (1 == src->stride[3]) && (1 == tensor->stride[3]) && THTensor_(nElement)(src) == THTensor_(nElement)(tensor))
  {
    real *tp = THTensor_(data)(src);
    real *rp = THTensor_(data)(tensor);
    long long sz = THTensor_(nElement)(tensor);
    long long srcBasicIndex = 0;
    
    long i = 0;
    long long k = 0;
    long long ii = 0;
    long long jj = 0;
    long long kk = 0;
    real *rpLocal = NULL;
    real *tpLocal = NULL;
  #pragma omp parallel for private(k)
    for(k=0; k < sz; k+=tensor->stride[2])
    { 
      kk = k/tensor->stride[0];
      jj = (k%tensor->stride[0])/tensor->stride[1];
      ii = (k%tensor->stride[1])/tensor->stride[2];
      srcBasicIndex = kk*src->stride[0] + jj*src->stride[1] + ii*src->stride[2];
      rpLocal = rp+k;
      tpLocal = tp+srcBasicIndex;
      #pragma ivdep
      for(i=0; i < tensor->stride[2]; i++)
      {
        rpLocal[i] = tpLocal[i];
      }
    }
  }
  else
  {
    TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = (real)(*src_data);)
  }
}

#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
}

IMPLEMENT_THTensor_COPY(Byte, unsigned char)
IMPLEMENT_THTensor_COPY(Char, char)
IMPLEMENT_THTensor_COPY(Short, short)
IMPLEMENT_THTensor_COPY(Int, int)
IMPLEMENT_THTensor_COPY(Long, long)
IMPLEMENT_THTensor_COPY(Float, float)
IMPLEMENT_THTensor_COPY(Double, double)

#endif
