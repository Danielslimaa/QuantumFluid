#pragma once
#include <math.h>
#include <cmath>
#include <omp.h>


 
double energy_per_particle_CPU(double* g, double* S, double *h_potential) {
 
    // Expression (4.6) A. Fabrocini et all, page 82.

    long double energy;

    long double part_1 = 0.0;
    long double part_2 = 0.0;
    long double part_3 = 0.0;
    long double aux1 = 0.0;
    long double aux2 = 0.0;

    int i;
#pragma omp parallel for
    for (i = 0; i < N; ++i){
        part_1 += (long double)(i) * g[i] * h_potential[i];
    }
    part_1 = rho * 3.14159265359 * U * dr * dr * part_1; // 3.14159265359 = 2*pi/2
#pragma omp parallel for
    for(i = 0; i < N; ++i){
        part_2 += (long double)(i) * (long double)(i) * (long double)(i) * (S[i]-1.0) * (S[i]-1.0) * (S[i]-1.0)/(S[i]);
    }
    part_2 = (-0.01989436788/rho) * dk * dk * dk * dk * part_2; // 0.01989436788 = 1/(2*4*2*pi)

#pragma omp parallel for
    for (i = 0; i < N-7; ++i){
        //part_3 += ( ((long double)(i)) * (3.0*(g[i+1]) + (g[i-1]) - (g[i+1]*g[i+1]/g[i]) - g[i] * 3.0) + ((g[i+1]) - g[i] * 1.0))/ (dr * dr);
        aux1 = ((-147.0*g[i] +360.0*g[i+1] -450.0*g[i+2] + 400.0*g[i+3]\
		-225.0*g[i+4] + 72.0*g[i+5] - 10.0*g[i+6]) / (60.0));
        aux2 = (812.0*g[i+0]-3132.0*g[i+1]+5265.0*g[i+2]\
        -5080.0*g[i+3]+2970.0*g[i+4]-972.0*g[i+5]+137.0*g[i+6])/(180);
        part_3 = (-((long double)i)/g[i])*aux1*aux1 + ((long double)i)*aux2; 
    }
    for (i = 7; i < N-2; ++i){
        part_3 += ( ((long double)(i)) * (3.0*(g[i+1]) + (g[i-1]) - (g[i+1]*g[i+1]/g[i]) - g[i] * 3.0) + ((g[i+1]) - g[i] * 1.0))/ (dr * dr);
     }




    //long double aux1, aux2;
    //#pragma omp parallel for
    //for (i = 1; i < N-1; ++i){
    //aux1 = (((g[i+1])+(g[i-1]) - 2.0*(g[i]))/(dr*dr) + ( (g[i+1] - g[i])/dr )/(long double(i)*dr));
    //aux2 = ( (g[i+1] - g[i])/dr );
    // part_30 += long double(i) * (aux1 - (aux2*aux2)/g[i]);
    //}


    //part_3 += g[0] * ((g[0+1]/g[0]) - 1.0)/ (dr * dr);
    
    part_3 += g[N-1] * ( ((long double)(N-1)) * (3.0*(1.0/g[N-1]) + (g[N-2]/g[N-1]) - (1.0/g[N-1])*(1.0/g[N-1]) - 3.0) + ((1.0/g[N-1]) - 1.0))/ (dr * dr);
    
    //part_3 = part_3 + (N-1) * g[N-1] * ((logl(1.0)+logl(fabsl(g[N-2])) - 2.0*logl(fabsl(g[N-1])))/(dr*dr) \
        + ( (logl(1.0)- logl(fabsl(g[N-1])))/dr )/((long double)(N-1)*dr)   );

    part_3 = (-0.78539816339*rho) * dr * dr * part_3; //0.78539816339 = 2*pi/(2*4)

    //part_30 = part_30 + ( (g[0+1] - g[0])/dr );

    //part_30 = part_30 + (N-1) * g[N-1] * ((logl(1.0)+logl(g[N-2]) - 2.0*logl(g[N-1]))/(dr*dr) \
   //     + ( (logl(1.0)- logl(g[N-1]))/dr )/(long double(N-1)*dr)   );

    //part_30 = (-0.78539816339*rho) * dr * dr * part_30; //0.78539816339 = 2*pi/(2*4)

    energy = part_1+part_2+part_3;

    return (double)energy;
}
