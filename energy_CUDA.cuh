#pragma once
#include <math.h>
#include <cmath>
#include <omp.h>
#include "kernels.cuh"
 
__host__ double energy_per_particle_CUDA(double *d_g, double *d_S, double *d_part1, double *d_part2, double *d_part3) {
  
  int BlocksPerGrid = 2; int ThreadsPerBlock = 1;
  // Expression (4.6) A. Fabrocini et all, page 82.
  double *energy = new double[1];
	energy_part1<<<BlocksPerGrid, ThreadsPerBlock>>>(d_g, d_part1);
  cudaDeviceSynchronize();
	energy_part2<<<BlocksPerGrid, ThreadsPerBlock>>>(d_S, d_part2);
  cudaDeviceSynchronize();
	energy_part3<<<BlocksPerGrid, ThreadsPerBlock>>>(d_g, d_part3);
  cudaDeviceSynchronize();
  *energy = 0;
  double *aux = new double[1]; 
  cudaMemcpy(energy, d_part1, sizeof(double), cudaMemcpyDeviceToHost);
  *energy += *aux;
  cudaMemcpy(aux, d_part2, sizeof(double), cudaMemcpyDeviceToHost);
  *energy += *aux;
  cudaMemcpy(aux, d_part3, sizeof(double), cudaMemcpyDeviceToHost);
  *energy += *aux;
  delete[] aux;
  return *energy;
}
