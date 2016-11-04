#include <mkl_vsl.h>
#include <sys/time.h>

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.c"
#else

#define min(x,y)  ( x<y?x:y )

void THTensor_(random)(THTensor *self, THGenerator *_generator)
{
#if defined(TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY(real, self, *self_data = (unsigned char)(THRandom_random(_generator) % (UCHAR_MAX+1)););
#elif defined(TH_REAL_IS_CHAR)
  TH_TENSOR_APPLY(real, self, *self_data = (char)(THRandom_random(_generator) % (CHAR_MAX+1)););
#elif defined(TH_REAL_IS_SHORT)
  TH_TENSOR_APPLY(real, self, *self_data = (short)(THRandom_random(_generator) % (SHRT_MAX+1)););
#elif defined(TH_REAL_IS_INT)
  TH_TENSOR_APPLY(real, self, *self_data = (int)(THRandom_random(_generator) % (INT_MAX+1UL)););
#elif defined(TH_REAL_IS_LONG)
  TH_TENSOR_APPLY(real, self, *self_data = (long)(THRandom_random(_generator) % (LONG_MAX+1UL)););
#elif defined(TH_REAL_IS_FLOAT)
  TH_TENSOR_APPLY(real, self, *self_data = (float)(THRandom_random(_generator) % ((1UL << FLT_MANT_DIG)+1)););
#elif defined(TH_REAL_IS_DOUBLE)
  TH_TENSOR_APPLY(real, self, *self_data = (double)(THRandom_random(_generator) % ((1ULL << DBL_MANT_DIG)+1)););
#else
#error "Unknown type"
#endif
}

void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_geometric(_generator, p););
}

void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p)
{
  THArgCheck(p >= 0 && p <= 1, 1, "must be >= 0 and <= 1");

  //time as initial seed
  struct timeval start;
  gettimeofday(&start,NULL);
  long seed = start.tv_sec * 1000 + (double)start.tv_usec/1000;
  
  //generate mt19937 random number
  VSLStreamStatePtr stream_mt19937;
  vslNewStream( &stream_mt19937, VSL_BRNG_MT19937, seed);
  int *seed_mt19937 = (int*)malloc(1*sizeof(int));
  vsRngUniform( VSL_RNG_METHOD_UNIFORM_STD, stream_mt19937, 1, seed_mt19937, 1, 4294967295);
  vslDeleteStream(&stream_mt19937);
 
  //mt19937 as seed to generate bernoulli  
  int n = THTensor_(nElement)(self);
  real *r = THTensor_(data)(self);
  int nthr = omp_get_max_threads();

  int *tmp = (int*)malloc(n*sizeof(int));

  # pragma omp parallel num_threads(nthr)
  {
    const int ithr = omp_get_thread_num();
    const int avg_amount = (n + nthr - 1) / nthr;
    const int my_offset = ithr * avg_amount;
    const int my_amount = min(my_offset + avg_amount, n) - my_offset;
         
    if (my_amount > 0) {
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, seed_mt19937[0]);
      vslSkipAheadStream(stream, my_offset);
      viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount,
        tmp + my_offset, p);
      vslDeleteStream(&stream);
    }
  }
  int k;
  for(k=0;k<n;k++)
  {
    r[k]=tmp[k];
  }
  free(tmp);
  free(seed_mt19937);
}

void THTensor_(bernoulli_FloatTensor)(THTensor *self, THGenerator *_generator, THFloatTensor *p)
{
  TH_TENSOR_APPLY2(real, self, float, p, *self_data = (real)THRandom_bernoulli(_generator, (double)*p_data););
}

void THTensor_(bernoulli_DoubleTensor)(THTensor *self, THGenerator *_generator, THDoubleTensor *p)
{
  TH_TENSOR_APPLY2(real, self, double, p, *self_data = (real)THRandom_bernoulli(_generator, (double)*p_data););
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

void THTensor_(uniform)(THTensor *self, THGenerator *_generator, double a, double b)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_uniform(_generator, a, b););
}

void THTensor_(normal)(THTensor *self, THGenerator *_generator, double mean, double stdv)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_normal(_generator, mean, stdv););
}

void THTensor_(exponential)(THTensor *self, THGenerator *_generator, double lambda)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_exponential(_generator, lambda););
}

void THTensor_(cauchy)(THTensor *self, THGenerator *_generator, double median, double sigma)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_cauchy(_generator, median, sigma););
}

void THTensor_(logNormal)(THTensor *self, THGenerator *_generator, double mean, double stdv)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_logNormal(_generator, mean, stdv););
}

void THTensor_(multinomial)(THLongTensor *self, THGenerator *_generator, THTensor *prob_dist, int n_sample, int with_replacement)
{
  int start_dim = THTensor_(nDimension)(prob_dist);
  long n_dist;
  long n_categories;
  THDoubleTensor* cum_dist;
  int i,j,k;

  if (start_dim == 1)
  {
    THTensor_(resize2d)(prob_dist, 1, THTensor_(size)(prob_dist, 0));
  }

  n_dist = THTensor_(size)(prob_dist, 0);
  n_categories = THTensor_(size)(prob_dist, 1);

  THArgCheck(n_sample > 0, 2, "cannot sample n_sample < 0 samples");

  if (!with_replacement)
  {
    THArgCheck((!with_replacement) && (n_sample <= n_categories), 2, \
    "cannot sample n_sample > prob_dist:size(1) samples without replacement");
  }

  /* cumulative probability distribution vector */
  cum_dist = THDoubleTensor_newWithSize1d(n_categories);

  /* will contain multinomial samples (category indices to be returned) */
  THLongTensor_resize2d(self, n_dist , n_sample);

  for (i=0; i<n_dist; i++)
  {
    /* Get normalized cumulative distribution from prob distribution */
    double sum = 0;
    for (j=0; j<n_categories; j++)
    {
      sum += THStorage_(get)( \
        prob_dist->storage, \
        prob_dist->storageOffset+i*prob_dist->stride[0]+j*prob_dist->stride[1] \
      );
      THDoubleStorage_set(
        cum_dist->storage, \
        cum_dist->storageOffset+j*cum_dist->stride[0], \
        sum \
      );
    }
    THArgCheckWithCleanup((sum > 0), THCleanup(THDoubleTensor_free(cum_dist);), 2,
                          "invalid multinomial distribution (sum of probabilities <= 0)");
    /* normalize cumulative probability distribution so that last val is 1
    i.e. dosen't assume original prob_dist row sums to one */
    if ( (sum > 0) || ( ( sum < 1.00001) && (sum > 0.99999) ) )
    {
      for (j=0; j<n_categories; j++)
      {
        THDoubleTensor_data(cum_dist)[j*cum_dist->stride[0]] /= sum;
      }
    }

    for (j=0; j<n_sample; j++)
    {
      /* sample a probability mass from a uniform distribution */
      double uniform_sample = THRandom_uniform(_generator, 0, 1);
      /* Do a binary search for the slot in which the prob falls
      ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
      int left_pointer = 0;
      int right_pointer = n_categories;
      int mid_pointer;
      double cum_prob;
      int sample_idx;
      /* Make sure the last cumulative distribution bucket sums to 1 */
      THDoubleTensor_data(cum_dist)[(n_categories-1)*cum_dist->stride[0]] = 1;

      while(right_pointer - left_pointer > 0)
      {
          mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
          cum_prob = THDoubleStorage_get( \
            cum_dist->storage, \
            cum_dist->storageOffset+mid_pointer*cum_dist->stride[0] \
          );
          if (cum_prob < uniform_sample)
          {
            left_pointer = mid_pointer + 1;
          }
          else
          {
            right_pointer = mid_pointer;
          }
      }
      sample_idx = left_pointer;

       /* store in result tensor (will be incremented for lua compat by wrapper) */
      THLongStorage_set( \
        self->storage, \
        self->storageOffset+i*self->stride[0]+j*self->stride[1], \
        sample_idx \
      );

      /* Once a sample is drawn, it cannot be drawn again. ie sample without replacement */
      if (!with_replacement)
      {
        /* update cumulative distribution so that sample cannot be drawn again */
        double diff;
        double new_val = 0;
        double sum;

        if (sample_idx != 0)
        {
          new_val = THDoubleStorage_get( \
            cum_dist->storage, \
            cum_dist->storageOffset+(sample_idx-1)*cum_dist->stride[0] \
          );
        }
        /* marginal cumulative mass (i.e. original probability) of sample */
        diff = THDoubleStorage_get( \
          cum_dist->storage, \
          cum_dist->storageOffset+sample_idx*cum_dist->stride[0] \
        ) - new_val;
        /* new sum of marginals is not one anymore... */
        sum = 1.0 - diff;
        for (k=0; k<n_categories; k++)
        {
          new_val = THDoubleStorage_get( \
            cum_dist->storage, \
            cum_dist->storageOffset+k*cum_dist->stride[0] \
          );
          if (k >= sample_idx)
          {
            /* remove sampled probability mass from later cumulative probabilities */
            new_val -= diff;
          }
          /* make total marginals sum to one */
          new_val /= sum;
          THDoubleStorage_set( \
            cum_dist->storage, \
            cum_dist->storageOffset+k*cum_dist->stride[0], \
            new_val \
          );
        }
      }
    }
  }

  THDoubleTensor_free(cum_dist);

  if (start_dim == 1)
  {
    THLongTensor_resize1d(self, n_sample);
    THTensor_(resize1d)(prob_dist, n_categories);
  }
}

#endif

#if defined(TH_REAL_IS_BYTE)
void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self)
{
  static const size_t size = sizeof(THGenerator);
  THGenerator *rng_state;
  THTensor_(resize1d)(self, size);
  THArgCheck(THTensor_(nElement)(self) == size, 1, "RNG state is wrong size");
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");
  rng_state = (THGenerator *)THTensor_(data)(self);
  THGenerator_copy(rng_state, _generator);
}

void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self)
{
  static const size_t size = sizeof(THGenerator);
  THGenerator *rng_state;
  THArgCheck(THTensor_(nElement)(self) == size, 1, "RNG state is wrong size");
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");
  rng_state = (THGenerator *)THTensor_(data)(self);
  THArgCheck(THGenerator_isValid(rng_state), 1, "Invalid RNG state");
  THGenerator_copy(_generator, rng_state);
}
#endif

#endif
