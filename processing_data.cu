#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <string>
#include <math.h>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <omp.h>
#include <stdio.h>

#include "kernels_PD.cuh"
#include "innitializer_PD.cuh"

int main(void)
{

	dr = 0.01 / 2.0;
	dk = 0.01 / 2.0;
	N = (1 << 13);

	//The potential v(r)

	double *h_potential = new double[N];

	for (int i = 0; i < N; ++i)
	{
		double r = ((double)i) * dr;
		h_potential[i] = exp(-r * r);
	}

	double *h_g = new double[N];
	double *h_S = new double[N];
	double *h_v_effective_r = new double[N];
	double *h_v_effective_k = new double[N];
	double *h_special_j0table = new double[N * N];
	double *d_potential, *d_g, *d_S, *d_v_effective_r, *d_v_effective_k, *d_laplaciano_N,
		*d_laplaciano_ln_g, *d_special_j0table, *d_N_r, *d_gradiente_g, *d_gradiente_N, *d_gradiente_ln_g;
	double *h_segundo_termo = new double[N];

	double *h_OBDM = new double[N];
	double *d_OBDM;

	double *h_n0 = new double[1];
	double *d_n0;

	calculate_J_matrix(h_special_j0table);

	U = 0.0; //rho = 0.2;

	int max_linha_U = 1;
	int max_coluna_rho = 1;
	int total = max_linha_U * max_coluna_rho;

	int k = 1;
	for (int linha_U = 0; linha_U < max_linha_U; ++linha_U)
	{

		rho = 0.1;

		for (int coluna_rho = 0; coluna_rho < max_coluna_rho; ++coluna_rho)
		{

			innitializer(d_n0, d_OBDM, h_potential, d_potential, d_g, d_S, d_v_effective_r, d_v_effective_k, d_laplaciano_N, d_laplaciano_ln_g,
						 d_special_j0table, h_special_j0table, d_N_r, d_gradiente_g, d_gradiente_N, d_gradiente_ln_g);

			read_g_and_S(h_g, d_g, h_S, d_S);

			//Cálculo do gradiente de g(r)
			calculo_gradiente_g<<<N / 1024, 1024>>>(d_g, d_gradiente_g);
			cudaDeviceSynchronize();

			//Cálculo do gradiente de log(g(r))
			calculo_gradiente_ln_g<<<N / 1024, 1024>>>(d_g, d_gradiente_g, d_gradiente_ln_g);
			cudaDeviceSynchronize();

			//Cálculo do laplaciano de N(r)
			calculate_laplaciano_Nodal<<<N / 1024, 1024>>>(d_S, d_special_j0table, d_laplaciano_N);
			cudaDeviceSynchronize();

			//Cálculo do terceiro termo, o que tem laplaciano
			calculo_laplaciano_ln_g<<<N / 1024, 1024>>>(d_g, d_laplaciano_ln_g);
			cudaDeviceSynchronize();

			calculo_Veff_r<<<N / 1024, 1024>>>(d_g, d_potential, d_laplaciano_N, d_laplaciano_ln_g, d_v_effective_r);
			cudaDeviceSynchronize();

			//CALCULA N(r) pela transformada de Fourier de N(k)
			//calculate_Nodal_r<<<N/1024, 1024>>>(d_S, d_special_j0table, d_N_r);
			cudaDeviceSynchronize();

			//CALCULA gradiente de N(r)
			//calculo_gradiente_Nodal<<<N/1024, 1024>>>(d_g, d_N_r, d_gradiente_N);
			cudaDeviceSynchronize();

			//ALTERNATIVA
			//calculo_Veff_r2<<<N/1024, 1024>>>(d_g, d_potential, d_gradiente_N, d_gradiente_g, d_gradiente_ln_g, d_v_effective_r);
			cudaDeviceSynchronize();

			calculate_Veff_k<<<N / 1024, 1024>>>(d_special_j0table, d_v_effective_r, d_v_effective_k);
			cudaDeviceSynchronize();

			compute_Nodal<<<N / 1024, 1024>>>(d_special_j0table, d_S, d_N_r);
			cudaDeviceSynchronize();

			calculate_n0<<<N / 1024, 1024>>>(d_g, d_N_r, d_n0);
			cudaDeviceSynchronize();

			calculate_OBDM<<<N / 1024, 1024>>>(d_special_j0table, d_N_r, d_n0, d_OBDM);
			cudaDeviceSynchronize();

			print_Veff(h_v_effective_k, d_v_effective_k);

			print_OBDM(h_OBDM, d_OBDM);

			cudaMemcpy(h_n0, d_n0, 1 * sizeof(double), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			std::cout << "(" << k << "/" << total << ")  "
					  << " --- (" << U << ", " << rho << ")"
					  << " -----> 0.5 * rho * v_eff[k = 0] = " << 0.5 * h_v_effective_k[0] << "  -----> n0 = " << h_n0[0] <<"\n"
					  << std::flush;
			//printf("\n 0.5 * rho * v_eff(k = 0) = %f\n", 0.5 * rho * h_v_effective_k[0]);

			rho = rho + 0.01;
			k = k + 1;

			free_Memory_GPU(d_potential, d_g, d_S, d_v_effective_r, d_v_effective_k, d_laplaciano_N, d_laplaciano_ln_g,
							d_special_j0table, d_N_r, d_gradiente_g, d_gradiente_N, d_gradiente_ln_g);
		}
		U = U + 0.1;
	}

	delete[] h_g;
	delete[] h_S;
	delete[] h_OBDM;
	delete[] h_v_effective_r;
	cudaDeviceReset();
	return 0;
}
