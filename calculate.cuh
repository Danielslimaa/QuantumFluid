#pragma once
#include "main_loop-X.cuh"
#include "auxiliary_functions.cuh"
#include <fstream>
#include <iostream>

__host__ void calculate(double *h_potential, double *&d_potential, double *h_g, double *&d_g, \
double *&h_S, double *&d_S, double *&d_first_term, double *&d_second_term, double *&d_third_term, \
double *&d_V_ph_k, double *&h_j0table, double *&d_j0table, double *&d_part1, double *&d_part2, double *&d_part3,\
double *energy_pp, double *diff_energy, double *energy, double *time_reference,\
int *k, double *derivative){

    char buffer [50000]; 

    int answer; 
    int okay = 0;

    while(okay == 0){

    innitialize_device(h_potential, d_potential, h_g, d_g, h_S, d_S, d_first_term, d_second_term, d_third_term, d_V_ph_k, h_j0table, d_j0table, d_part1, d_part2, d_part3);

    answer = calculate_one_scenario(h_potential, d_potential, derivative, k, h_g, h_S, d_g, d_S, d_j0table, d_V_ph_k, energy_pp,\
                            diff_energy, d_first_term, d_second_term, d_third_term,\
                            buffer, energy, time_reference, d_part1, d_part2, d_part3);
    
    free_memory_and_reset_device(d_potential, d_g, d_V_ph_k, d_S, d_j0table, d_first_term, d_second_term, d_third_term, d_part1,\
        d_part2, d_part3);

    if(answer == 1){okay = 1;}
    
    if(answer == 2){ 
        printf("\n ----- Error: g(r) < 0 at time = %.6f s. |diff. energies| = %.3e. Derivative = %.3e. Continuing with dt divided by 2. \n ", (double) (k[0]+1) * dt, diff_energy[k[0]], *derivative);
        *time_reference = (double) (k[0]+1) * dt;
        dt = dt / 2.0;

        printf("Parameters:\n U = %.0f, rho = %.3f, precision = %2.1e, and dt = %.10f.  \n\n",  U, rho, precision, dt); okay = 0;

    }
    
    if(answer == 3){okay = 1;}
    if(answer == 4){
    printf("\n ----- Error: energy < 0 at time = %.6f s. |diff. energies| = %.3e. Derivative = %.3e. Continuing with dt divided by 2. \n ", (double) (k[0]+1) * dt, diff_energy[k[0]], *derivative);
    *time_reference = (double) (k[0]+1) * dt;
    dt = dt / 2.0;

    printf("Parameters:\n U = %.0f, rho = %.3f, precision = %2.1e, and dt = %.10f.  \n\n",  U, rho, precision, dt); okay = 0;
    }

    if(answer == 5){printf("The number of interactions approached the limit M."); okay = 1;}

    char log_buffer[300];

    sprintf(log_buffer, "LOG_2U100-0.1rho2.5.dat");

    std::ofstream myfile;
    myfile.open(log_buffer, std::ios::app);
    myfile << U << "	" << rho << "	"  << answer << "\n"; 
    myfile.close();

    }

    

}
