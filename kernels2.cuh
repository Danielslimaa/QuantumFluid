#pragma once
#include <math.h>
#include <cmath>
#include <cuda_runtime.h>

__device__ __managed__ double U;
__device__ __managed__ double rho;
__device__ __managed__ double dt;
__device__ __managed__ double precision;
__device__ __managed__ double dr;
__device__ __managed__ double dk;
__device__ __managed__ int N;

