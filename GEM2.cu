#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <omp.h>
#include <stdio.h>
#include <cstdlib>

#include "main_loop-X.cuh"
#include "auxiliary_functions.cuh"
#include "calculate.cuh"

int Blocks_N;
int ThreadsPerBlock_N;
int M = 30000; //5376;

int main(void)
{

  dr = 0.01/2.0;
  dk = 0.01/2.0;
  N = (1 << 13);

  //The potential v(r)

  double *h_potential = new double[N];

  for (int i = 0; i < N; ++i)
  {
    double r = ((double)i) * dr;
    h_potential[i] = exp(-pow(r,2));
  }

  if (N >= 1024)
  {
    ThreadsPerBlock_N = 1024;
  }
  else
  {
    ThreadsPerBlock_N = N;
  }
  Blocks_N = (int)ceil((double)N / 1024.0);

  std::cout << "Blocks_N: " << Blocks_N << std::endl;
  std::cout << "ThreadsPerBlock_N: " << ThreadsPerBlock_N << std::endl;

  double *h_g = new double[N];
  double *h_S = new double[N];
  double *h_j0table = new double[N * N];
  double *energy_pp = new double[M];
  double *derivative = new double[1];
  double *diff_energy = new double[M];
  double *energy = new double[1];
  double *d_g, *d_j0table, *d_first_term, *d_second_term, *d_third_term;
  double *d_V_ph_k, *d_S;
  double *d_part1, *d_part2, *d_part3, *d_potential;
  time_t start, start_total, end_total;
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
  double *d_aux_j0table;
  cudaMalloc(&d_aux_j0table, N * N * sizeof(double));
  BesselJ0Table<<<numBlocks, threadsPerBlock>>>(d_aux_j0table);
  cudaDeviceSynchronize();
  cudaMemcpy(h_j0table, d_aux_j0table, N * N * sizeof(double), cudaMemcpyDeviceToHost);
  //CPU_BesselJ0Table(h_j0table);

  int *reference = new int[1];
  double *time_reference = new double[1];
  int *k = new int[1];

  U = 0;
  double initial_rho = 0.1;
  rho = initial_rho;
  double initial_dt = 0.025;
  dt = initial_dt;

  double initial_precision = 2.00e-5;
  precision = initial_precision;

  int linha_U = 0;
  int coluna_rho = 0;
  int max_linha_U = 1;
  int max_coluna_rho = 1;
  double *vector_dt = new double[max_coluna_rho];
  int total = max_linha_U * max_coluna_rho;
  int aux_total = 1;

  for (int i = 0; i < max_coluna_rho; ++i)
  {
    vector_dt[i] = initial_dt;
  }

  printf("State:\n N = %d, dr = dk = %f, dt = %f, and precision = %1.0e. \n\n", N, dr, dt, precision);
  time(&start_total);

  for (linha_U = 0; linha_U < max_linha_U; ++linha_U)
  {
    rho = initial_rho;

    for (coluna_rho = 0; coluna_rho < max_coluna_rho; ++coluna_rho)
    {
      dt = initial_dt; // vector_dt[coluna_rho];
      time(&start);

      calculate(h_potential, d_potential, h_g, d_g, h_S, d_S, d_first_term, d_second_term, d_third_term,
                d_V_ph_k, h_j0table, d_j0table, d_part1, d_part2, d_part3, energy_pp, diff_energy, energy,
                time_reference, k, derivative);
      vector_dt[coluna_rho] = dt;
      exit_results(h_g, h_S, energy_pp, diff_energy, k, dt, aux_total, total, start);

      rho = rho + 0.01;
      ++aux_total;
    }
    U = U + 0.1;
  }

  std::cout << "\n\nTotal computational time-lapse: " << time(&end_total) - start_total << " s." << std::endl;

  delete[] vector_dt;
  delete[] h_potential;
  delete[] energy_pp;
  delete[] diff_energy;
  delete[] derivative;
  delete[] h_g;
  delete[] h_S;
  delete[] h_j0table;
  delete[] energy;
  delete[] k;
  delete[] reference;
  delete[] time_reference;
  return 0;
}
