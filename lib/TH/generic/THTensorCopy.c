#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else
#define TH_OMP_THRESHOLD 10
void THTensor_(copy)(THTensor *tensor, THTensor *src)
{

int dimI = src->nDimension;
int dimO = tensor->nDimension;
//printf("THTensor_(copy) src %s\n", (THTensor_(isContiguous)(src))? "ture":"false");
//printf("THTensor_(copy) dst %s\n", (THTensor_(isContiguous)(tensor))? "ture":"false");

if (THTensor_(isContiguous)(src) && THTensor_(isContiguous)(tensor) && THTensor_(nElement)(src) == THTensor_(nElement)(tensor))
  {
    real *tp = THTensor_(data)(src);
    real *rp = THTensor_(data)(tensor);
    long sz = THTensor_(nElement)(src);
    long j;
    real z;
//    printf("NB parallel THTensor_(copy)\n");
    #pragma omp parallel for if(sz > TH_OMP_THRESHOLD) private(j)
    for (j=0; j<sz; j++)
    {
      rp[j] = tp[j];
    }
  }
  else if((dimI==dimO) && (dimI==3) && THTensor_(isContiguous)(tensor) && THTensor_(nElement)(src) == THTensor_(nElement)(tensor))
  {
//     struct timeval timestamp1, timestamp2;
//      int time_cost = 0;
//     gettimeofday(&timestamp1, NULL);
      real *tp = THTensor_(data)(src);
      real *rp = THTensor_(data)(tensor);
      long sz = THTensor_(nElement)(tensor);
      long long srcBasicIndex = 0;
      
      long i = 0;
      long long k = 0;
      long jj = 0;
      long  kk = 0;
#pragma omp parallel for private(k)
      for(k=0; k < sz; k+=tensor->stride[1])
      { 
        kk = k/tensor->stride[0];
        jj = (k%tensor->stride[0])/tensor->stride[1];
        srcBasicIndex = kk*src->stride[0] + jj*src->stride[1];
        for(i=0; i < tensor->stride[1]; i++)
        {
          rp[i+k] = tp[i+srcBasicIndex];
        }
      }
  }
  else if((dimI==dimO) && (dimI == 4) && THTensor_(isContiguous)(tensor) && THTensor_(nElement)(src) == THTensor_(nElement)(tensor))
  {
      real *tp = THTensor_(data)(src);
      real *rp = THTensor_(data)(tensor);
      long sz = THTensor_(nElement)(tensor);
      long long srcBasicIndex = 0;
      
      long i = 0;
      long long k = 0;
      long ii = 0;
      long jj = 0;
      long  kk = 0;
#pragma omp parallel for private(k)
      for(k=0; k < sz; k+=tensor->stride[2])
      { 
        kk = k/tensor->stride[0];
        jj = (k%tensor->stride[0])/tensor->stride[1];
        ii = (k%tensor->stride[1])/tensor->stride[2];
        srcBasicIndex = kk*src->stride[0] + jj*src->stride[1] + ii*src->stride[2];
        for(i=0; i < tensor->stride[2]; i++)
        {
          rp[i+k] = tp[i+srcBasicIndex];
        }
      }
  }

  else if((dimI==dimO) && (dimI == 4) && THTensor_(isContiguous)(src) && THTensor_(nElement)(src) == THTensor_(nElement)(tensor))
  {
      real *tp = THTensor_(data)(src);
      real *rp = THTensor_(data)(tensor);
      long sz = THTensor_(nElement)(tensor);
      long long dstBasicIndex = 0;
      
      long i = 0;
      long long k = 0;
      long ii = 0;
      long jj = 0;
      long  kk = 0;
#pragma omp parallel for private(k)
      for(k=0; k < sz; k+=src->stride[2])
      { 
        kk = k/src->stride[0];
        jj = (k%src->stride[0])/src->stride[1];
        ii = (k%src->stride[1])/src->stride[2];
        dstBasicIndex = kk*tensor->stride[0] + jj*tensor->stride[1] + ii*tensor->stride[2];
        for(i=0; i < src->stride[2]; i++)
        {
          rp[i+dstBasicIndex] = tp[i+k];
        }
      }
  }

  else
  {
    //printf("SB serial THTensor_(copy)\n");
    printf("SB serial THTensor_(copy), dimI=%d, dimO=%d, srcisContiguous=%d, dstisContiguous=%d\n",dimI,dimO,THTensor_(isContiguous)(src),THTensor_(isContiguous)(tensor));
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
