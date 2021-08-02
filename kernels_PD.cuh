#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include <fstream>
#include <iostream>

__device__ __managed__ double U;
__device__ __managed__ double rho;
__device__ __managed__ double dt;
__device__ __managed__ double precision;
__device__ __managed__ double dr;
__device__ __managed__ double dk;
__device__ __managed__ int N;

__global__ void BesselJ0Table(double *d_j0table)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_x = blockDim.x * gridDim.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_y = blockDim.y * gridDim.y;
	int i, j;
	for (i = index_x; i < N; i += stride_x)
	{
		for (j = index_y; j < N; j += stride_y)
		{
			d_j0table[i * N + j] = j0f(double(i) * dr * double(j) * dk);
		}
	}
}

void CPU_BesselJ0Table(double *h_j0table)
{
	int i, j;
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			h_j0table[i * N + j] = j0f((double)(i)*dr * (double)(j)*dk);
		}
	}
}

void printer_table(double *matrix, const char *name)
{
	std::ofstream myfile;
	myfile.open(name);
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			myfile << matrix[i * N + j] << " ";
		}
		std::cout << "\n";
	}
	myfile.close();
	return;
}

void printer_vector(double *vetor, int n, const char *name)
{
	std::ofstream myfile;
	myfile.open(name);
	for (int i = 0; i < n; ++i)
	{
		myfile << vetor[i] << "\n";
	}
	myfile.close();
	return;
}

void load_vector(double *vetor, int n, const char *name)
{
	std::ifstream myfile;
	myfile.open(name);
	for (int i = 0; i < n; ++i)
	{
		myfile >> vetor[i];
	}
	myfile.close();
}

__global__ void calculo_gradiente_g(double *d_g, double *d_gradiente_g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if (i >= 0 && i < N - 5)
	{

		aux = (-137. * d_g[i + 0] + 300. * d_g[i + 1] - 300. * d_g[i + 2] + 200. * d_g[i + 3] - 75. * d_g[i + 4] + 12. * d_g[i + 5]) / (60.0);

		d_gradiente_g[i] = aux / dr;
	}
	if (i > N - 6 && i < N)
	{
		d_gradiente_g[i] = (d_g[i] - d_g[i - 1]) / dr;
	}
}

__global__ void calculo_gradiente_ln_g(double *d_g, double *d_gradiente_g, double *d_gradiente_ln_g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		d_gradiente_ln_g[i] = d_gradiente_g[i] * d_gradiente_g[i] / d_g[i];
	}
}

__global__ void calculate_laplaciano_Nodal(double *d_S, double *kernel, double *d_N_r)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double sum;
	if (i < N)
	{
		sum = 0.0;
		for (int j = 0; j < N; ++j)
		{
			sum += ((double)j) * ((double)j) * ((double)j) * kernel[i * N + j] * (d_S[j] - 1) * (d_S[j] - 1.0) / d_S[j];
		}
		d_N_r[i] = sum * dk * dk * dk * dk / (6.28318530718 * rho);
	}
}

__global__ void calculo_gradiente_Nodal(double *d_g, double *d_N_r, double *d_gradiente_Nodal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if (i >= 0 && i < N - 5)
	{

		aux = (-137. * d_N_r[i + 0] + 300. * d_N_r[i + 1] - 300. * d_N_r[i + 2] + 200. * d_N_r[i + 3] - 75. * d_N_r[i + 4] + 12. * d_N_r[i + 5]) / (60.0);

		d_gradiente_Nodal[i] = aux / dr;
	}
	if (i > N - 6 && i < N)
	{
		d_gradiente_Nodal[i] = (d_N_r[i] - d_N_r[i - 1]) / dr;
	}
}

__global__ void calculo_laplaciano_ln_g(double *d_g, double *d_laplaciano_ln_g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double aux1, aux2;
	if (i > 0 && i < N - 7)
	{

		aux1 = ((-147.0 * d_g[i] + 360.0 * d_g[i + 1] - 450.0 * d_g[i + 2] + 400.0 * d_g[i + 3] - 225.0 * d_g[i + 4] + 72.0 * d_g[i + 5] - 10.0 * d_g[i + 6]) / (60.0));

		aux2 = (812.0 * d_g[i + 0] - 3132.0 * d_g[i + 1] + 5265.0 * d_g[i + 2] - 5080.0 * d_g[i + 3] + 2970.0 * d_g[i + 4] - 972.0 * d_g[i + 5] + 137.0 * d_g[i + 6]) / (180);

		d_laplaciano_ln_g[i] = ((-aux1 * aux1 / (d_g[i] * d_g[i])) + (aux2 / d_g[i]) + (aux1 / (d_g[i] * (double)i))) / (dr * dr);
	}
	if (i > N - 8 && i < N - 2)
	{
		d_laplaciano_ln_g[i] = ((3.0 * (d_g[i + 1]) + (d_g[i - 1]) - (d_g[i + 1] * d_g[i + 1] / d_g[i]) - d_g[i] * 3.0) + ((d_g[i + 1]) - d_g[i] * 1.0)) / (d_g[i] * dr * dr);
	}
	if (i > N - 3 && i < N - 1)
	{
		d_laplaciano_ln_g[i] = ((3.0 * (d_g[i + 1]) + (d_g[i - 1]) -
								 (d_g[i + 1] * d_g[i + 1] / d_g[i]) - d_g[i] * 3.0) +
								((d_g[i + 1]) - d_g[i] * 1.0)) /
							   (d_g[i] * dr * dr);
	}
	if (i == 0)
	{
		d_laplaciano_ln_g[i] = d_laplaciano_ln_g[1];
	}
}

__global__ void calculo_Veff_r(double *d_g, double *d_potential, double *d_laplaciano_Nodal, double *d_laplaciano_ln_g, double *d_v_effective_r)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		d_v_effective_r[i] = d_g[i] * U * d_potential[i] + (-0.25) * d_g[i] * d_laplaciano_ln_g[i] + (-0.25) * d_g[i] * d_laplaciano_Nodal[i];
	}
}

__global__ void calculo_Veff_r2(double *d_g, double *d_potential, double *d_gradiente_Nodal, double *d_gradiente_g, double *d_gradiente_ln_g, double *d_v_effective_r)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		d_v_effective_r[i] = d_g[i] * U * d_potential[i] + (0.25) * (d_gradiente_g[i] * d_gradiente_g[i] / d_g[i]) + (-0.25) * d_gradiente_g[i] * d_gradiente_Nodal[i];
	}
}

__global__ void calculate_Veff_k(double *d_special_j0table, double *d_v_effective_r, double *v_effective_k)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	double aux;
	if (row < N)
	{
		aux = 0.0;
		for (int i = 0; i < N; i++)
		{
			aux += (double)(i)*d_special_j0table[row * N + i] * (d_v_effective_r[i]);
		}
		v_effective_k[row] = aux * dr * dr * 6.28318530718 * rho;
	}
}

__global__ void compute_Nodal(double *d_special_j0table, double *d_S, double *d_N_r)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double sum;
	if (i < N)
	{
		sum = 0.0;
		for (int j = 0; j < N; ++j)
		{
			sum += ((double)j) * d_special_j0table[i * N + j] * (d_S[j] - 1) * (d_S[j] - 1.0) / d_S[j];
		}
		d_N_r[i] = sum * dk * dk / (6.28318530718 * rho);
	}
}

__global__ void calculate_OBDM(double *d_special_j0table, double *d_N_r, double *d_n0, double *d_OBDM)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		d_OBDM[i] = d_n0[0] * exp(d_N_r[i]) / rho;
	}
}

__global__ void calculate_n0(double *d_g, double *d_N_r, double *d_n0)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0.0;
	if (i < N)
	{
		sum += ((double)i) * (d_g[i] + 1.0 + d_N_r[i]) - 0.5 * ((double)i) * (d_g[i] + 1.0) * d_N_r[i];
		//sum -= 0.5 * ((double)i) * (d_g[i] + 1) * d_N_r[i]; 
	}
	d_n0[0] = exp(sum * 6.28318530718 * rho * dr * dr);
}