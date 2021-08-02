#pragma once

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <bits/stdc++.h>

#include "kernels.cuh"

__host__ void innitialize_device(double *h_potential, double *&d_potential, double *h_g, double *&d_g, \
double *&h_S, double *&d_S, double *&d_first_term, double *&d_second_term, double *&d_third_term, \
double *&d_V_ph_k, double *&h_j0table, double *&d_j0table, double *&d_part1, double *&d_part2, double *&d_part3)
{

	cudaMalloc(&d_potential, N * sizeof(double));
  cudaMalloc(&d_g, N * sizeof(double));
  cudaMalloc(&d_S, N * sizeof(double));
  cudaMalloc(&d_first_term, N * sizeof(double));
  cudaMalloc(&d_second_term, N * sizeof(double));
  cudaMalloc(&d_third_term, N * sizeof(double));
  cudaMalloc(&d_V_ph_k, N * sizeof(double));
  cudaMalloc(&d_j0table, N * N * sizeof(double));
  cudaMalloc(&d_part1, sizeof(double));
  cudaMalloc(&d_part2, sizeof(double));
  cudaMalloc(&d_part3, sizeof(double));

  cudaMemcpy(d_potential, h_potential, N * sizeof(double), cudaMemcpyHostToDevice); //Allocating the potential in the device
  cudaMemcpy(d_j0table, h_j0table, N * N * sizeof(double), cudaMemcpyHostToDevice); //Allocating the j0table in the device
  
  
  
}

__host__ void free_memory_and_reset_device(double *&d_potential, double *&d_g, double *&d_V_ph_k, double *&d_S, double *&d_j0table,\
 double *&d_first_term, double *&d_second_term, double *&d_third_term, double *&d_part1, double *&d_part2, double *&d_part3)
{

  // Free memory
  cudaFree(d_potential);
  cudaFree(d_g);
  cudaFree(d_V_ph_k);
  cudaFree(d_S);
  cudaFree(d_j0table);
  cudaFree(d_first_term);cudaFree(d_second_term);cudaFree(d_third_term);
  cudaFree(d_part1);cudaFree(d_part2);cudaFree(d_part3);
  cudaDeviceReset();

}

__host__ void exit_results(double *h_g, double *h_S, double *energy_pp, double *diff_energy, int *k, double dt,\
 int aux_total, int total, time_t &start)
{
  time_t end; 
  char buffer [50000]; char g_buffer [50000]; char S_buffer [50000]; char epp_buffer [50000]; char file_buffer [50000];
	
  sprintf(g_buffer, "U-500-rho-2/g-U-%.1f-rho-%1.3f.dat", U, rho);//"2g-functions-5U20-0.4rho2.0/g-U-%.1f-rho-%1.3f.dat", U, rho);
  sprintf(S_buffer, "U-500-rho-2/S-U-%.1f-rho-%1.3f.dat", U, rho);//"2S-functions-5U20-0.4rho2.0/S-U-%.1f-rho-%1.3f.dat", U, rho);
  sprintf(epp_buffer, "U-500-rho-2//epp-U-%.0f-rho-%1.3f.dat", U, rho);
  sprintf(buffer, "U = %3.1f, rho = %1.3f. e = %.8f +- %.3e. dt = %1.10f.", U, rho, energy_pp[k[0]-1], diff_energy[k[0]-1], dt);
  sprintf(file_buffer, "U-500-rho-2/energia-U-500-rho-2.000.dat");
  
  if(aux_total < 2){
    std::ofstream myfile;
    myfile.open(file_buffer); 
    myfile.close();
  }
               
  std::cout << "\r(" << aux_total<<"/" << total << ") ---- " << buffer << " Computational time-lapse: " <<\
  time(&end)-start << " s. Number of iteractions = " << k[0] << ". Time of evolution = " << (double)k[0]*dt << " s" <<  std::endl;
            
  printer_vector(h_g, N, g_buffer);
  printer_vector(h_S, N, S_buffer);
  printer_vector(energy_pp, k[0], epp_buffer);
  
  std::ofstream myfile;
  myfile.open(file_buffer, std::ios::app);
  myfile << U << "	" << rho << "	"  << energy_pp[k[0]] << "\n"; 
  myfile.close();


}
