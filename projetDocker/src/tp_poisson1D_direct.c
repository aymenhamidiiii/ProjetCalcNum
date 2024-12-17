/******************************************/
/* tp2_poisson1D_direct.c                 */
/* This file contains the main function   */
/* to solve the Poisson 1D problem        */
/******************************************/
#include "lib_poisson1D.h"
#include <time.h>

#define TRF 0
#define TRI 1
#define SV 2

int main(int argc,char *argv[]) {
    int ierr;
    int jj;
    int nbpoints, la;
    int ku, kl, kv, lab;
    int *ipiv;
    int info = 1;
    int NRHS;
    int IMPLEM = 0;
    double T0, T1;
    double *RHS, *EX_SOL, *X;
    double **AAB;
    double *AB;
    double relres;
    clock_t start, end; // Chronomètre
    double cpu_time_used;

    if (argc == 2) {
        IMPLEM = atoi(argv[1]);
    } else if (argc > 2) {
        perror("Application takes at most one argument");
        exit(1);
    }

    NRHS=1;
    nbpoints=10;
    la=nbpoints-2;
    T0=-5.0;
    T1=5.0;

    printf("--------- Poisson 1D ---------\n\n");
    RHS=(double *) malloc(sizeof(double)*la);
    EX_SOL=(double *) malloc(sizeof(double)*la);
    X=(double *) malloc(sizeof(double)*la);

    set_grid_points_1D(X, &la);
    set_dense_RHS_DBC_1D(RHS,&la,&T0,&T1);
    set_analytical_solution_DBC_1D(EX_SOL, X, &la, &T0, &T1);

    kv=1;
    ku=1;
    kl=1;
    lab=kv+kl+ku+1;

    AB = (double *) malloc(sizeof(double)*lab*la);
    if (AB == NULL) {
        perror("Allocation error for AB");
        exit(1);
    }

    set_GB_operator_colMajor_poisson1D(AB, &lab, &la, &kv);
  //   double *x = (double *)malloc(sizeof(double) * la);
  //   double *y = (double *)calloc(la, sizeof(double));
  //   double *y_ref = (double *)calloc(la, sizeof(double));

  //   // Initialisation de x
  //   for (int i = 0; i < la; i++) {
  //       x[i] = i + 1.0; // x = [1, 2, 3, ..., la]
  //   }

  //   // Validation de dgbmv
  //   validate_dgbmv(AB, x, y, y_ref, &lab, &la, &kl, &ku);

  //   // Libération de la mémoire
  // free(x);
  // free(y);
  // free(y_ref);



    ipiv = (int *) calloc(la, sizeof(int));
    if (ipiv == NULL) {
        perror("Allocation error for ipiv");
        exit(1);
    }

    /* Start the timer */
    start = clock();

    /* LU Factorization */
    if (IMPLEM == TRF) {
        dgbtrf_(&la, &la, &kl, &ku, AB, &lab, ipiv, &info);
        printf("INFO after dgbtrf_: %d\n", info);
    }

    /* LU for tridiagonal matrix */
    if (IMPLEM == TRI) {
        dgbtrftridiag(&la, &la, &kl, &ku, AB, &lab, ipiv, &info);
    }

    /* Solve the system */
    if (IMPLEM == TRI || IMPLEM == TRF) {
        if (info == 0) {
            dgbtrs_("N", &la, &kl, &ku, &NRHS, AB, &lab, ipiv, RHS, &la, &info);
        }
    }

    /* Solve with dgbsv */
    if (IMPLEM == SV) {
        dgbsv_(&la, &kl, &ku, &NRHS, AB, &lab, ipiv, RHS, &la, &info);
        if (info != 0) {
            printf("\nINFO DGBSV = %d\n", info);
        }
    }

    /* Stop the timer */
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    /* Relative forward error */
    relres = relative_forward_error(RHS, EX_SOL, &la);

    printf("\nThe relative forward error is relres = %e\n", relres);
    printf("Execution time: %f seconds\n", cpu_time_used);

    /* Display theoretical complexity */
    if (IMPLEM == TRF) {
        printf("Theoretical Complexity of dgbtrf: O(n * (kl + ku)^2)\n");
    } else if (IMPLEM == TRI) {
        printf("Theoretical Complexity of custom LU: O(n)\n");
    } else if (IMPLEM == SV) {
        printf("Theoretical Complexity of dgbsv: O(n * (kl + ku)^2)\n");
    }

    free(RHS);
    free(EX_SOL);
    free(X);
    free(AB);
    free(ipiv);
    printf("\n\n--------- End -----------\n");
}
