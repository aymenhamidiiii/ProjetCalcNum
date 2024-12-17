/**********************************************/
/* lib_poisson1D_richardson.c                 */
/* Numerical library developed to solve 1D    */ 
/* Poisson problem (Heat equation)            */
/**********************************************/
#include "lib_poisson1D.h"
#include <math.h>
#include <stdlib.h>
#include <cblas.h>
#include <lapacke.h> // Inclut LAPACKE

/* Fonction pour récupérer les valeurs propres maximales et minimales */
double eigmax_poisson1D(int *la) {
    return 4.0; // Valeur propre maximale de la matrice de Poisson
}

double eigmin_poisson1D(int *la) {
    double sin_term = sin(M_PI / (2 * (*la + 1)));
    return 4.0 * sin_term * sin_term; // Approximation de la valeur propre minimale
}

double richardson_alpha_opt(int *la) {
    double eigmax = eigmax_poisson1D(la);
    double eigmin = eigmin_poisson1D(la);
    return 2.0 / (eigmax + eigmin);
}

void richardson_alpha(double *AB, double *RHS, double *X, double *alpha_rich, int *lab, 
                      int *la, int *ku, int *kl, double *tol, int *maxit, 
                      double *resvec, int *nbite) {
    int k = 0;
    double res_norm, res_old, *residual;

    residual = (double *)malloc((*la) * sizeof(double));

    // Initialisation : X initial = 0
    for (int i = 0; i < *la; i++) X[i] = 0.0;

    // Résidu initial : r_0 = b - Ax_0
    cblas_dgbmv(CblasColMajor, CblasNoTrans, *la, *la, *kl, *ku, -1.0, AB, *lab, X, 1, 1.0, RHS, 1);
    cblas_dcopy(*la, RHS, 1, residual, 1); // r_0 = RHS
    res_old = cblas_dnrm2(*la, residual, 1);
    resvec[0] = res_old;

    while (k < *maxit && res_old > *tol) {
        // x_k+1 = x_k + alpha * r_k
        cblas_daxpy(*la, *alpha_rich, residual, 1, X, 1);

        // r_k+1 = b - Ax_k+1
        cblas_dgbmv(CblasColMajor, CblasNoTrans, *la, *la, *kl, *ku, -1.0, AB, *lab, X, 1, 1.0, RHS, 1);
        res_old = cblas_dnrm2(*la, RHS, 1);
        resvec[k + 1] = res_old;

        k++;
    }

    *nbite = k;
    free(residual);
}

void extract_MB_jacobi_tridiag(double *AB, double *MB, int *lab, int *la, int *ku, int *kl, int *kv) {
    for (int j = 0; j < *la; j++) {
        for (int i = 0; i < *lab; i++) {
            MB[i + j * (*lab)] = 0.0; // Initialisation
        }
        MB[*kv + j * (*lab)] = AB[*kv + j * (*lab)]; // Diagonale principale
    }
}

void extract_MB_gauss_seidel_tridiag(double *AB, double *MB, int *lab, int *la, int *ku, int *kl, int *kv) {
    for (int j = 0; j < *la; j++) {
        for (int i = 0; i < *lab; i++) {
            MB[i + j * (*lab)] = 0.0; // Initialisation
        }
        MB[*kv + j * (*lab)] = AB[*kv + j * (*lab)]; // Diagonale principale
        if (j > 0) MB[*kv - 1 + j * (*lab)] = AB[*kv - 1 + j * (*lab)]; // Sous-diagonale
    }
}

void richardson_MB(double *AB, double *RHS, double *X, double *MB, int *lab, int *la, 
                   int *ku, int *kl, double *tol, int *maxit, double *resvec, int *nbite) {
    int k = 0;
    double res_norm, res_old, *residual;

    residual = (double *)malloc((*la) * sizeof(double));

    // Initialisation : X initial = 0
    for (int i = 0; i < *la; i++) X[i] = 0.0;

    // Résidu initial : r_0 = b - Ax_0
    cblas_dgbmv(CblasColMajor, CblasNoTrans, *la, *la, *kl, *ku, -1.0, AB, *lab, X, 1, 1.0, RHS, 1);
    cblas_dcopy(*la, RHS, 1, residual, 1);
    res_old = cblas_dnrm2(*la, residual, 1);
    resvec[0] = res_old;

    while (k < *maxit && res_old > *tol) {
        // r_k = M^{-1} * r_k
        cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, *la, MB, *lab, residual, 1);

        // x_k+1 = x_k + r_k
        cblas_daxpy(*la, 1.0, residual, 1, X, 1);

        // r_k+1 = b - Ax_k+1
        cblas_dgbmv(CblasColMajor, CblasNoTrans, *la, *la, *kl, *ku, -1.0, AB, *lab, X, 1, 1.0, RHS, 1);
        res_old = cblas_dnrm2(*la, RHS, 1);
        resvec[k + 1] = res_old;

        k++;
    }

    *nbite = k;
    free(residual);
}
