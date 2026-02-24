/* Phase function Legendre expansion — ported from 6SV2.1 TRUNCA.f
 * Decomposes ctx->polar.pha[] into ctx->polar.betal[] coefficients.
 * Polarization (gammal, deltal, alphal, zetal) not yet implemented. */
#include "../include/sixs_ctx.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "gauss.h"

/* Expand ctx->polar.pha[] into Legendre coefficients ctx->polar.betal[].
 * Always sets *coeff = 0 (no truncation in this 6SV2.1 version).
 * ipol: 0 = scalar only; nonzero = also compute gammal/deltal (not implemented). */
void sixs_trunca(SixsCtx *ctx, int ipol, float *coeff)
{
    (void)ipol;  /* polarization expansion not implemented */

    int nbmu   = ctx->quad.nquad;   /* e.g. 83 */
    int nbmu_2 = (nbmu - 3) / 2;   /* 40 for nbmu=83 */

    /* Set up Gauss quadrature with special endpoints at -1, 0, +1 */
    float cgaus_S[NQ_MAX];
    float pdgs_S[NQ_MAX];
    sixs_gauss_setup(nbmu, cgaus_S, pdgs_S);

    /* Zero out Legendre coefficients */
    int ncoeff = nbmu - 2;   /* 0 .. nbmu-3 */
    for (int k = 0; k < ncoeff; k++) ctx->polar.betal[k] = 0.0f;

    /* Compute betal[k] = (2k+1)/2 * sum_j pha[j] * w[j] * P_k(cos_j) */
    /* Use double precision to accumulate; pl uses offset pointer trick */
    double pl_arr[NQ_MAX + 2];   /* pl_arr[k+1] = P_k */
    double *pl = pl_arr + 1;     /* so pl[-1], pl[0], pl[k] work */

    for (int j = 0; j < nbmu; j++) {
        double x  = (double)ctx->polar.pha[j] * (double)pdgs_S[j];
        double rm = (double)cgaus_S[j];
        pl[-1] = 0.0;
        pl[0]  = 1.0;
        for (int k = 0; k < ncoeff; k++) {
            pl[k+1] = ((2*k + 1.0) * rm * pl[k] - k * pl[k-1]) / (k + 1.0);
            ctx->polar.betal[k] += (float)(x * pl[k]);
        }
    }

    /* Normalise and zero out negative coefficients */
    for (int k = 0; k < ncoeff; k++) {
        ctx->polar.betal[k] = (float)((2*k + 1.0) * 0.5 * ctx->polar.betal[k]);
        if (ctx->polar.betal[k] < 0.0f) {
            for (int j = k; j < ncoeff; j++) ctx->polar.betal[j] = 0.0f;
            break;
        }
    }

    *coeff = 0.0f;   /* no truncation applied */
    (void)nbmu_2;
}
