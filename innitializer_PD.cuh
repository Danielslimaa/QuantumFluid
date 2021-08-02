#pragma once
#include <cuda_runtime.h>

#include "kernels_PD.cuh"

void innitializer(double *&d_n0, double *&d_OBDM, double *h_potential, double *&d_potential, double *&d_g, double *&d_S, double *&d_v_effective_r, double *&d_v_effective_k,
                  double *&d_segundo_termo, double *&d_terceiro_termo, double *&d_special_j0table, double *h_special_j0table, double *&d_N_r,
                  double *&d_gradiente_g, double *&d_gradiente_N, double *&d_gradiente_ln_g)
{

    cudaMalloc(&d_n0, 1 * sizeof(double));
    cudaMalloc(&d_OBDM, N * sizeof(double));
    cudaMalloc(&d_g, N * sizeof(double));
    cudaMalloc(&d_potential, N * sizeof(double));

    cudaMemcpy(d_potential, h_potential, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_S, N * sizeof(double));
    cudaMalloc(&d_v_effective_r, N * sizeof(double));
    cudaMalloc(&d_v_effective_k, N * sizeof(double));
    cudaMalloc(&d_segundo_termo, N * sizeof(double));
    cudaMalloc(&d_terceiro_termo, N * sizeof(double));
    cudaMalloc(&d_special_j0table, N * N * sizeof(double));

    cudaMemcpy(d_special_j0table, h_special_j0table, N * N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_N_r, N * sizeof(double));
    cudaMalloc(&d_gradiente_g, N * sizeof(double));
    cudaMalloc(&d_gradiente_N, N * sizeof(double));
    cudaMalloc(&d_gradiente_ln_g, N * sizeof(double));
}

void free_Memory_GPU(double *&d_potential, double *&d_g, double *&d_S, double *&d_v_effective_r, double *&d_v_effective_k,
                     double *&d_segundo_termo, double *&d_terceiro_termo, double *&d_special_j0table, double *&d_N_r,
                     double *&d_gradiente_g, double *&d_gradiente_N, double *&d_gradiente_ln_g)
{

    cudaFree(d_potential);
    cudaFree(d_g);
    cudaFree(d_S);
    cudaFree(d_v_effective_r);
    cudaFree(d_v_effective_k);
    cudaFree(d_segundo_termo);
    cudaFree(d_terceiro_termo);
    cudaFree(d_special_j0table);
    cudaFree(d_N_r);
    cudaFree(d_gradiente_g);
    cudaFree(d_gradiente_N);
    cudaFree(d_gradiente_ln_g);
    cudaDeviceReset();
}

void calculate_J_matrix(double *&h_special_j0table)
{
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    double *d_aux_j0table;
    cudaMalloc(&d_aux_j0table, N * N * sizeof(double));
    BesselJ0Table<<<numBlocks, threadsPerBlock>>>(d_aux_j0table);
    cudaDeviceSynchronize();
    cudaMemcpy(h_special_j0table, d_aux_j0table, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    //std::cout << "h_special_j0table[100000] = " << h_special_j0table[100000] << std::endl;
    printf("Initialization: OKAY \n");
    printf("N x N Bessel table calculation: OKAY \n");
    //printf("h_special_j0table[100000] = %f\n", h_special_j0table[100000]);
    cudaFree(d_aux_j0table);
}

void read_g_and_S(double *&h_g, double *&d_g, double *&h_S, double *&d_S)
{

    char file_g_buffer[20000];
    char file_S_buffer[20000];

    sprintf(file_g_buffer, "U-500-rho-2/g-U-%.1f-rho-%.3f.dat", U, rho);
    sprintf(file_S_buffer, "U-500-rho-2/S-U-%.1f-rho-%.3f.dat", U, rho);

    load_vector(h_g, N, file_g_buffer);
    // h_g[0] = 0.8*h_g[5];h_g[1] = 0.85*h_g[5];h_g[2] = 0.9*h_g[5];h_g[4] = 0.95*h_g[5];
    cudaMemcpy(d_g, h_g, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    load_vector(h_S, N, file_S_buffer);
    // h_S[0] = 0.8*h_S[5];h_S[1] = 0.85*h_S[5];h_S[2] = 0.9*h_S[5];h_S[4] = 0.95*h_S[5];
    cudaMemcpy(d_S, h_S, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void print_Veff(double *&h_v_effective_k, double *&d_v_effective_k)
{
    char file_Veff_k_buffer[20000];
    sprintf(file_Veff_k_buffer, "Veff-5U25-0.4rho2.0/v_effective_k_U-%.1f-rho-%.3f.dat", U, rho);
    cudaMemcpy(h_v_effective_k, d_v_effective_k, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printer_vector(h_v_effective_k, N, file_Veff_k_buffer);
}

void print_OBDM(double *&h_OBDM, double *&d_OBDM)
{
    char file_OBDM_buffer[20000];
    sprintf(file_OBDM_buffer, "OBDM/OBDM-U-%.1f-rho-%.3f.dat", U, rho);
    cudaMemcpy(h_OBDM, d_OBDM, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    double aux = h_OBDM[0];
    for (int i = 0; i < N; ++i)
    {
        h_OBDM[i] = h_OBDM[i] / aux;
    }
    printer_vector(h_OBDM, N, file_OBDM_buffer);
}