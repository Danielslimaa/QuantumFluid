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

__global__ void energy_part1( double *d_g, double *d_part1)
{
double sum = 0.0; double x;

for (int i = 0; 
i < N; 
i += blockDim.x)
{
    x = (double) (i + threadIdx.x);
    sum += x * d_g[i + threadIdx.x] * expf(-x * dr * x * dr);
}
 *d_part1 = rho * 3.14159265359 * U * dr * dr * sum; // 3.14159265359 = 2*pi/2
}

__global__ void energy_part2( double *S, double *d_part2)
{
double sum = 0.0; double x;
for (int i = 0; 
i < N; 
i += blockDim.x)
{
    x = (double) (i + threadIdx.x);
    sum += x*x*x*(S[i + threadIdx.x]-1.0)*(S[i + threadIdx.x]-1.0)*(S[i + threadIdx.x]-1.0)/(S[i + threadIdx.x]);
 }
 *d_part2 = (-0.01989436788/rho) * sum * dk * dk * dk * dk; // 0.01989436788 = 1/(2*4*2*pi)
}

__global__ void energy_part3( double *g, double *d_part3)
{

double sum = 0.0; 
for (int i = 1; 
i < N - 1; 
i += blockDim.x)
{
    sum += g[i + threadIdx.x] * ((double)(i + threadIdx.x)*(log(fabs(g[i + threadIdx.x+1]))+log(fabs(g[i + threadIdx.x-1]))\
                                          - 2.0*log(fabs(g[i + threadIdx.x])))/(dr*dr)\
        + ( (log(fabs(g[i + threadIdx.x+1])) - log(fabs(g[i + threadIdx.x])))/dr )/(dr));
 }
sum = sum + g[0] * ( (log(fabs(g[0+1])) - log(fabs(g[0])))/dr );
sum = sum + (N-1) * g[N-1] * ((log(1.0)+log(fabs(g[N-2])) - 2.0*log(fabs(g[N-1])))/(dr*dr) \
        + ( (log(1.0)- log(fabs(g[N-1])))/dr )/((double)(N-1)*dr)   );

*d_part3 = (-0.78539816339*rho) * dr * dr * sum; //0.78539816339 = 2*pi/(2*4)
}

__global__ void energy_total( double *d_part1, double *d_part2, double *d_part3, double *energy)
{
    *energy = *d_part1 + *d_part2 + *d_part3; 
}

__global__ void BesselJ0Table(double* d_j0table)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_x = blockDim.x * gridDim.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_y = blockDim.y * gridDim.y;
	int i, j;
	for (i = index_x; i < N; i += stride_x) {
		for (j = index_y; j < N; j += stride_y) {
			d_j0table[i * N + j] = j0f(double(i) * dr * double(j) * dk);
		}
	}
}

__global__ void g_from_S(double* d_g, double* d_S, double* d_j0table)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if(col < N){
		aux = 0.0;
		for (int i = 0; i < N; i++) {
			aux += double(i) * d_j0table[i * N + col] * (d_S[i] - 1.0);
		}
		d_g[col] = 1.0 + aux * dr * dr / (rho * 6.28318530718);
	}
}

__global__ void S_from_g(double* d_g, double* d_S, double* d_j0table)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if(row < N){
		for (int i = 0; i < N; i++) {
			aux += double(i) * d_j0table[row * N + i] * (d_g[i] - 1.0);
		}
		d_S[row] = 1.0 + (rho * 6.28318530718) * aux * dk * dk;
	}
}

__global__ void calculate_first_term(double* d_g, double* d_first_term, double *d_potential) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N) {
		d_first_term[i] = U * d_g[i] * d_potential[i];
	}
}

__global__ void calculate_second_term(double* d_g, double* d_second_term)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if (i < N-7) {
        aux = ((-147.0*d_g[i] +360.0*d_g[i+1] -450.0*d_g[i+2] + 400.0*d_g[i+3]\
		-225.0*d_g[i+4] + 72.0*d_g[i+5] - 10.0*d_g[i+6]) / (60.0*dr));
        d_second_term[i] = aux * aux / (4.0 * d_g[i]);
	}
	if(i > N - 6){
		aux = ((d_g[1] - d_g[0]) / dr);
		d_second_term[i] = aux * aux / (4.0 * d_g[i]);// g(r) is even, then its derivative is odd
	}
}

__global__ void calculate_second_term2(double *d_g, double *d_second_term)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double aux1, aux2;

	if (i < N - 7)
	{
        aux1 = ((-147.0*d_g[i] +360.0*d_g[i+1] -450.0*d_g[i+2] + 400.0*d_g[i+3]\
		-225.0*d_g[i+4] + 72.0*d_g[i+5] - 10.0*d_g[i+6]) / (60.0));
        aux2 = (812.0*d_g[i+0]-3132.0*d_g[i+1]+5265.0*d_g[i+2]\
        -5080.0*d_g[i+3]+2970.0*d_g[i+4]-972.0*d_g[i+5]+137.0*d_g[i+6])/(180);
		d_second_term[i] = -0.25 * ((-((double)i)/d_g[i])*aux1*aux1 + ((double)i)*aux2) / (dr * dr);
	}
	if (i < N - 2)
	{
		d_second_term[i] += -0.25 * (((double)(i)) * (3.0 * (d_g[i + 1]) + (d_g[i - 1]) - (d_g[i + 1] * d_g[i + 1] / d_g[i]) - d_g[i] * 3.0) \
		+ ((d_g[i + 1]) - d_g[i] * 1.0)) / (dr * dr);
	}
	if (i == N -1)
	{
		d_second_term[i] = d_second_term[N-2]; 
	}
}

__global__ void calculate_third_term(double* d_g, double* d_S, double* d_j0table, \
	double* d_third_term)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	int j;
	aux = 0.0;

	for (j = 1; j < N; ++j) {
		aux += double(j) * d_j0table[j * N + col] * (-(0.25) * (2.0 * d_S[j] + 1.0) * (\
			(double(j) * dk) - (double(j) * dk) / (d_S[j])) * ((double(j) * dk) - (double(j) * dk)\
				/ (d_S[j])));
	}
	d_third_term[col] = (d_g[col] - 1.0) * aux * dk * dk / (rho * 6.28318530718);
}

__global__ void calculate_V_ph_k(double* d_first_term, double* d_second_term, \
	double* d_third_term, double* d_kernel, double* d_V_ph_k) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	double twopi = 6.283185307179586;
	int j;
	if(row < N){
		aux = 0.0;
		for (j = 0; j < N; ++j) {
			aux += double(j) * d_kernel[row * N + j] * (d_first_term[j] + \
				d_second_term[j] + d_third_term[j]);
		}
		d_V_ph_k[row] = aux * dr * dr * rho * twopi;
	}
}

__global__ void update_S(double* d_V_ph_k, double* d_S) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double epsilon = 1e-15;
	if(i < N) {
		d_S[i] = dt * (double(i) * dk + epsilon) / sqrtf(fabs((double(i) * dk + epsilon) * (double(i) * dk + epsilon) \
			+ 4.0 * d_V_ph_k[i])) + (1.0-dt) * d_S[i];
	}

}

__global__ void teste_update_S(double* d_V_ph_k, double* d_S) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N) {
		d_S[i] = dt * (double(i) * dk) / sqrtf((double(i) * dk) * (double(i) * dk) \
			+ 4.0 * d_V_ph_k[i]) + (1.0-dt) * d_S[i];
	}

}




