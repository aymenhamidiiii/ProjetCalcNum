/**********************************************/
/* lib_poisson1D.c                            */
/* Numerical library developed to solve 1D    */ 
/* Poisson problem (Heat equation)            */
/**********************************************/
#include "lib_poisson1D.h"
#include <math.h>
#include <cblas.h> // Inclure BLAS pour dgbmv

// void validate_dgbmv(double* AB, double* x, double* y, double* y_ref, int *lab, int *la, int *kl, int *ku) {
//     int i, j;
//     double alpha = 1.0, beta = 0.0;

//     // Produit matrice-vecteur avec dgbmv
//     cblas_dgbmv(CblasColMajor, CblasNoTrans, *la, *la, *kl, *ku, alpha, AB, *lab, x, 1, beta, y, 1);

//     // Calcul manuel du produit A*x (pour tridiagonale)
//     for (i = 0; i < *la; i++) {
//         y_ref[i] = 0.0;
//         if (i > 0) {
//             y_ref[i] += -1.0 * x[i - 1]; // Sous-diagonale
//         }
//         y_ref[i] += 2.0 * x[i]; // Diagonale principale
//         if (i < (*la - 1)) {
//             y_ref[i] += -1.0 * x[i + 1]; // Sur-diagonale
//         }
//     }

//     // Comparaison des résultats
//     printf("\nValidation of dgbmv:\n");
//     for (i = 0; i < *la; i++) {
//         printf("y[%d] = %f, y_ref[%d] = %f, diff = %e\n", i, y[i], i, y_ref[i], fabs(y[i] - y_ref[i]));
//     }
// }


void set_GB_operator_colMajor_poisson1D(double* AB, int *lab, int *la, int *kv) {
    int i, j;

    // Initialisation complète à 0.0
    for (j = 0; j < *la; j++) {
        for (i = 0; i < *lab; i++) {
            AB[i + j * (*lab)] = 0.0;
        }
    }

    // Remplissage de la matrice bandée
    for (j = 0; j < *la; j++) {
        if (j > 0) {
            AB[*kv - 1 + j * (*lab)] = -1.0; // Sous-diagonale
        }
        AB[*kv + j * (*lab)] = 2.0;           // Diagonale principale
        if (j < (*la - 1)) {
            AB[*kv + 1 + j * (*lab)] = -1.0; // Sur-diagonale
        }
    }
}



void set_GB_operator_colMajor_poisson1D_Id(double* AB, int *lab, int *la, int *kv){

      int i, j;

    for (j = 0; j < *la; j++) {
        for (i = 0; i < *lab; i++) {
            AB[i + j * (*lab)] = 0.0; // Initialise avec des 0
        }
        AB[*kv + j * (*lab)] = 1.0; // Diagonale principale = 1.0
    }
}

void set_dense_RHS_DBC_1D(double* RHS, int* la, double* BC0, double* BC1){
      int i;
    for (i = 0; i < *la; i++) {
        RHS[i] = 0.0; // Initialise RHS à 0
    }
    RHS[0] -= *BC0; // Condition au bord gauche
    RHS[*la - 1] -= *BC1; // Condition au bord droit
}  

void set_analytical_solution_DBC_1D(double* EX_SOL, double* X, int* la, double* BC0, double* BC1){
      int i;
    for (i = 0; i < *la; i++) {
        EX_SOL[i] = *BC0 + (X[i] * (*BC1 - *BC0)); // Solution analytique linéaire
    }
}  

void set_grid_points_1D(double* x, int* la){
    int i;
    double h = 1.0 / (*la + 1);
    for (i = 0; i < *la; i++) {
        x[i] = (i + 1) * h; // Points de grille uniformes
    }
}

double relative_forward_error(double* x, double* y, int* la) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < *la; i++) {
        num += (x[i] - y[i]) * (x[i] - y[i]);
        den += y[i] * y[i];
    }
    if (den == 0.0) {
        printf("Warning: Denominator is zero. Returning 0.\n");
        return 0.0; // Retourne 0 si y est nul
    }
    return sqrt(num) / sqrt(den);
}

int indexABCol(int i, int j, int *lab){
  return (*lab) * j + i;
}

int dgbtrftridiag(int *la, int*n, int *kl, int *ku, double *AB, int *lab, int *ipiv, int *info){
      int i;
    double l;

    for (i = 0; i < *n - 1; i++) {
        if (AB[*ku + i * (*lab)] == 0.0) {
            *info = i + 1; // Pivot nul, la factorisation échoue
            return *info;
        }

        l = AB[*ku - 1 + (i + 1) * (*lab)] / AB[*ku + i * (*lab)]; // Calcul du facteur
        AB[*ku - 1 + (i + 1) * (*lab)] = l; // Stocke L sous la diagonale principale
        AB[*ku + (i + 1) * (*lab)] -= l * AB[*ku + 1 + i * (*lab)]; // Mise à jour U
    }

    *info = 0; // Succès
  return *info;
}
