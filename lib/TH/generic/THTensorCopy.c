#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else
#define TH_OMP_THRESHOLD 10
#define TH_MAX_DIM 100       // temporary
void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
    int dimI = src->nDimension;
    int dimO = tensor->nDimension;

 //   long tensorSz = THTensor_(nElement)(tensor);
    int srcContg = THTensor_(isContiguous)(src)? 1:0;
    int tensorContg = THTensor_(isContiguous)(tensor)? 1:0;

    if ( srcContg && tensorContg )
    {
        real *tp = THTensor_(data)(src);
        real *rp = THTensor_(data)(tensor);
        long sz = THTensor_(nElement)(src);
        long j;
        #pragma omp parallel for if(sz > TH_OMP_THRESHOLD) private(j)
        #pragma ivdep
        for (j=0; j<sz; j++)
        {
          rp[j] = tp[j];
        }
    }
  else
  {
        int strideConI = 1;
        int strideConO = 1;
        int srcSizeStride[TH_MAX_DIM] = {0};
        int tensorSizeStride[TH_MAX_DIM] = {0};
        int dim = 0;
        int initMulSize = 1;
        for(dim = dimI; dim > 0; dim--)
        {
          initMulSize *= src->size[dim-1];
          srcSizeStride[dim-1] = initMulSize;
          if(0 == src->stride[dim])
          {
            strideConI = 0;
          }
        }

        initMulSize = 1;
        for(dim = dimO; dim > 0; dim--)
        {
          initMulSize *= tensor->size[dim-1];
          tensorSizeStride[dim-1] = initMulSize;
          if(0 == tensor->stride[dim])
          {
             strideConO = 0;
          }

        }
        if((strideConI != 0) && (strideConO != 0))
        {
          real *tp = THTensor_(data)(src);
          real *rp = THTensor_(data)(tensor);
          long sz = THTensor_(nElement)(src);
          if((dimI==dimO) && (dimI > 2) && tensorContg )
          {
            long srcBasicIndex = 0;
            long iter = 0;
            int dim = 0;
            long i = 0;
            long j = 0;
            long index = 0;

            real *rpLocal = NULL;
            real *tpLocal = NULL;
            #pragma omp parallel for    //there is no parallelism below this level
            for(iter=0; iter < sz; iter+=tensor->stride[dimI-2])  //not -1 to make use of vectorization
            {
                index = iter/tensor->stride[0];
                srcBasicIndex = index*src->stride[0];
                for(dim = 1; dim < dimI-1; dim++)
                {
                    index = (iter%tensor->stride[dim-1])/tensor->stride[dim];
                    srcBasicIndex += index*src->stride[dim];
                }

                rpLocal = rp+iter;
                tpLocal = tp+srcBasicIndex;
                j=0;
                #pragma ivdep
                for(i=0; i < tensor->stride[dimI-2]; i++)  // not contiguous requirement
                {
                  rpLocal[i] = tpLocal[j];
                  j+= src->stride[dimI-1];
                }
            }

          }
          else if((dimI==dimO) && (dimI > 2) && srcContg ) //
          {
            long tensorBasicIndex = 0;
            long iter = 0;
            int dim = 0;
            long i = 0;
            long j = 0;
            long index = 0;

            real *rpLocal = NULL;
            real *tpLocal = NULL;
            #pragma omp parallel for   //there is no parallelism below this level
            for(iter=0; iter < sz; iter+=src->stride[dimI-2])  //not -1 to make use of vectorization
            {
                index = iter/src->stride[0];
                tensorBasicIndex = index*tensor->stride[0];
                for(dim = 1; dim < dimI-1; dim++)
                {
                    index = (iter%src->stride[dim-1])/src->stride[dim];
                    tensorBasicIndex += index*tensor->stride[dim];
                }

                tpLocal = tp+iter;
                rpLocal = rp+tensorBasicIndex;
                j=0;
                #pragma ivdep
                for(i=0; i < src->stride[dimI-2]; i++)
                {
                  rpLocal[j] = tpLocal[i];
                  j+= tensor->stride[dimI-1];
                }
            }
          }
          else
          {
            long srcBasicIndex = 0;
            long tensorBasicIndex = 0;

            long i = 0;
            long j = 0;
            long index = 0;
            long iter = 0;
            #pragma omp parallel for   //there is no parallelism below this level
            for (iter = 0; iter < sz; iter++)
            {
                srcBasicIndex = 0;
                tensorBasicIndex = 0;

                for(dim = 0; dim < dimI-1; dim++)
                {
                    index = (iter%srcSizeStride[dim])/srcSizeStride[dim+1];
                    srcBasicIndex += index*src->stride[dim];
                }
                index = iter%srcSizeStride[dim];
                srcBasicIndex += index*src->stride[dim];

                for(dim = 0; dim < dimO-1; dim++)
                {
                    index = (iter%tensorSizeStride[dim])/tensorSizeStride[dim+1];
                    tensorBasicIndex += index*tensor->stride[dim];
                }
                index = iter%tensorSizeStride[dim];
                tensorBasicIndex += index*tensor->stride[dim];

                *(rp+tensorBasicIndex) = *(tp+srcBasicIndex);
            }
           }
        }
        else
        {
          TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = (real)(*src_data););
        }

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
