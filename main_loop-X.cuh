#pragma once
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
#include "energy_CPU.h"
#include "energy_CUDA.cuh"

extern int Blocks_N; 
extern int ThreadsPerBlock_N; 
extern int M;


void CPU_BesselJ0Table(double *j0table){
    int i, j;

    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N ; ++j)
        {
            j0table[i * N + j] = j0f( (double)(i)*dk*(double)(j)*dr);
        }
    }
    return;
}


void printer_vector(double* vetor, int n, const char* name) {
	std::ofstream myfile;
	myfile.open(name);
	for (int i = 0; i < n; ++i) {
		myfile << vetor[i] << "\n";
	}
	myfile.close();
	return;
}


int calculate_one_scenario(double *h_potential, double *d_potential, double *derivada, int *pk, double *h_g, double *h_S,
                            double *d_g, double *d_S, double *d_j0table, \
                            double *d_V_ph_k, double *energy_pp, double *diff_energy,\
                            double *d_first_term, double *d_second_term, \
                            double *d_third_term, char *const buffer, double *exit_energy, \
							double *time_reference, double *d_part1, double *d_part2, double *d_part3){
     
  
	for (int i = 0; i < N; ++i) {h_g[i] = 1.0;}
	cudaMemcpy(d_g, h_g, N * sizeof(double), cudaMemcpyHostToDevice);
	S_from_g << < Blocks_N, ThreadsPerBlock_N>> > (d_g, d_S, d_j0table);
	cudaDeviceSynchronize();
	double energy; double aux_energy = 0.0; 
	double real_time; 

  
	int trigger = 1;  int i; int teste = 0;
	int k = 0; char loop_buffer[5000];


    while (trigger == 1) {
	        
		//First term
		calculate_first_term << < Blocks_N, ThreadsPerBlock_N >> > (d_g, d_first_term, d_potential);
		cudaDeviceSynchronize();

		//Second term
		calculate_second_term << < Blocks_N, ThreadsPerBlock_N >> > (d_g, d_second_term);
		cudaDeviceSynchronize();

		//Third term
		calculate_third_term << < Blocks_N, ThreadsPerBlock_N >> > (d_g, d_S, d_j0table, \
			d_third_term);
		cudaDeviceSynchronize();

		//d_V_ph_k
		calculate_V_ph_k << < Blocks_N, ThreadsPerBlock_N >> > (d_first_term, \
			d_second_term, d_third_term, d_j0table, d_V_ph_k);
		cudaDeviceSynchronize();

		//Updating S(k)
		if(teste == 0){update_S << < Blocks_N, ThreadsPerBlock_N >> > (d_V_ph_k, d_S);}
		if(teste == 1){teste_update_S << < N / 1024, 1024 >> > (d_V_ph_k, d_S); teste = 2;}
		cudaDeviceSynchronize();

		//Updating g(r)
		g_from_S << < Blocks_N, ThreadsPerBlock_N >> > (d_g, d_S, d_j0table);
		cudaDeviceSynchronize();
		
		cudaMemcpy(h_g, d_g, N*sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaMemcpy(h_S, d_S, N*sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		for (i = 0; i < N; ++i){
			if(h_g[i] < 0.0){
				*time_reference = real_time; 
				trigger = 0;
   				return 2;
			}
		}

		real_time = ((double) k)*dt; //+ *time_reference;                                        	

		//*energy = energy_per_particle_CUDA(d_g, d_S, d_part1, d_part2, d_part3);
		energy = energy_per_particle_CPU(h_g, h_S, h_potential);
	
		if (std::isnan(energy) == 1) {
			//for (int m = 0; m < 3; ++m) { std::cout << ".x.x.x.x.x.x.x.x.x.x.x.x.x." << std::endl; }
			std::cout << "------ Calculation interrupted due to error in the energy calculation. ------ " << std::endl;
			printf("U = %3.1f e rho = %1.3f. Final value of energy = %3.8f +- %1.5e \n", U, rho, energy_pp[k-1],diff_energy[k - 1]);
			//for (int m = 0; m < 3; ++m) { std::cout << ".x.x.x.x.x.x.x.x.x.x.x.x.x." << std::endl; }
			trigger = 0;
			return 1;
		}
		    					
		if (energy < 0){return 4;}
		if (k == M - 1){return 5;}		

		energy_pp[k] = energy;
		diff_energy[k] = std::abs(energy - aux_energy);
		derivada[0] = diff_energy[k]/dt; 							    
		sprintf(loop_buffer, "\r Time = %.6f s -----------------> e = %.6f +- %1.3e.", real_time, energy, diff_energy[k]);					
		std::cout << loop_buffer << std::flush;		

		if ( derivada[0] < precision )  
		{			
			if(teste == 0){teste = 1;}
			if(teste == 2){
				*time_reference = 0;  
				trigger = 0;
				pk[0] = k;
				k++;
				*exit_energy = energy;
				return 3;
			}		
		}		
		aux_energy = energy;
		pk[0] = k;
		k++;
		exit_energy[0] = energy;			
	}
	return 3;
}
