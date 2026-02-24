/* Gauss-Legendre quadrature — ported from 6SV2.1 GAUSS.f */
#include "../include/sixs_ctx.h"
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void sixs_gauss(float x1, float x2, float *x, float *w, int n) {
    const double eps = 3.0e-14;
    int m = (n + 1) / 2;
    double xm = 0.5 * (x2 + x1);
    double xl = 0.5 * (x2 - x1);

    for (int i = 1; i <= m; i++) {
        double z = cos(M_PI * (i - 0.25) / (n + 0.5));
        double z1, p1, p2, p3, pp;
        do {
            p1 = 1.0; p2 = 0.0;
            for (int j = 1; j <= n; j++) {
                p3 = p2; p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;
        } while (fabs(z - z1) > eps);
        if (fabs(z) < eps) z = 0.0;
        x[i - 1]     = (float)(xm - xl * z);
        x[n - i]     = (float)(xm + xl * z);
        w[i - 1]     = (float)(2.0 * xl / ((1.0 - z * z) * pp * pp));
        w[n - i]     = w[i - 1];
    }
}

/* Compute the full Gauss quadrature set used by 6SV:
 * nquad-3 interior points plus -1, 0, +1 special points.
 * Fills cgaus_S[nquad] and pdgs_S[nquad]. */
void sixs_gauss_setup(int nquad, float *cgaus_S, float *pdgs_S) {
    int nbmu_2 = (nquad - 3) / 2;
    float cosang[NQ_MAX], weight[NQ_MAX];
    sixs_gauss(-1.0f, 1.0f, cosang, weight, nquad - 3);

    cgaus_S[0] = -1.0f; pdgs_S[0] = 0.0f;
    for (int j = 0; j < nbmu_2; j++) {
        cgaus_S[j + 1] = cosang[j];
        pdgs_S[j + 1]  = weight[j];
    }
    cgaus_S[nbmu_2 + 1] = 0.0f; pdgs_S[nbmu_2 + 1] = 0.0f;
    for (int j = nbmu_2; j < nquad - 3; j++) {
        cgaus_S[j + 2] = cosang[j];
        pdgs_S[j + 2]  = weight[j];
    }
    cgaus_S[nquad - 1] = 1.0f; pdgs_S[nquad - 1] = 0.0f;
}
