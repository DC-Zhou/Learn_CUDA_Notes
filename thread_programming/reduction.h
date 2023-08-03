#ifndef _REDUCTION_H_
#define _REDUCTION_H_

// @reduction_global_kernel.cu
void global_reduction(float *d_out, float *d_in, int n_threads, int size);

// @reduction_shared_kernel.cu
void shared_reduction(float *d_out, float *d_in, int n_threads, int size);

#endif // _REDUCTION_H_