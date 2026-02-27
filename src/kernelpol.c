/* Polarized scattering kernel — ported from 6SV2.1 KERNELPOL.f
 *
 * Computes generalized spherical functions psl, rsl, tsl and the Müller
 * matrix kernels (bp, arr, att, art) used by sixs_ospol().
 *
 * Simplification vs full 6SV2.1: aerosol treated as spherical particles
 * (gammal = zetal = 0, alphal = betal).  This means:
 *   gr(j,k) = gt(j,k) = 0          (no aerosol I↔Q,U coupling)
 *   arr(j,k) = Σ_l rsl(l,j)·rsl(l,k)·betal(l)
 *   att(j,k) = Σ_l tsl(l,j)·tsl(l,k)·betal(l)
 *   art(j,k) = Σ_l tsl(l,j)·rsl(l,k)·betal(l)
 *
 * The Rayleigh I↔Q,U coupling (gamma2 terms) is injected in-line by
 * sixs_ospol() and does NOT appear here. */

#include "kernelpol.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

void sixs_kernelpol(const SixsCtx *ctx, int is, int mu,
                    const float *rm_off,
                    double *xpl_off,
                    double *xrl_off,
                    double *xtl_off,
                    double *psl,
                    double *rsl,
                    double *tsl,
                    double *bp,
                    double *arr,
                    double *art,
                    double *att)
{
    int nquad = ctx->quad.nquad;
    int ip1   = nquad - 3;
    double rac3 = sqrt(3.0);
    int dim = 2 * mu + 1;

    /* Access macros — same layout as kernel.c */
    #define RM(j)      rm_off[(j)+mu]
    #define PSL(l,j)   psl[((l)+1)*dim + ((j)+mu)]
    #define RSL(l,j)   rsl[((l)+1)*dim + ((j)+mu)]
    #define TSL(l,j)   tsl[((l)+1)*dim + ((j)+mu)]
    #define XPL(j)     xpl_off[(j)+mu]
    #define XRL(j)     xrl_off[(j)+mu]
    #define XTL(j)     xtl_off[(j)+mu]
    #define BP(j,k)    bp [(j)*dim + ((k)+mu)]
    #define ARR(j,k)   arr[(j)*dim + ((k)+mu)]
    #define ART(j,k)   art[(j)*dim + ((k)+mu)]
    #define ATT(j,k)   att[(j)*dim + ((k)+mu)]

    /* ── Initialise generalised spherical functions at order l = is ── */

    if (is == 0) {
        /* Legendre polynomials: psl[0,j]=1, psl[1,j]=±c, psl[2,j]=(3c²−1)/2
         * Q-functions rsl: rsl[2,j] = 3(1−c²)/2/√6
         * U-functions tsl: tsl[2,j] = 0 for is=0 */
        for (int j = 0; j <= mu; j++) {
            double c = (double)RM(j);
            PSL(0,  j) = 1.0;  PSL(0, -j) = 1.0;
            PSL(1,  j) = c;    PSL(1, -j) = -c;
            double xdb = (3.0*c*c - 1.0) * 0.5;
            if (fabs(xdb) < 1e-30) xdb = 0.0;
            PSL(2,  j) = xdb;  PSL(2, -j) = xdb;

            RSL(1,  j) = 0.0;  RSL(1, -j) = 0.0;
            xdb = 3.0 * (1.0 - c*c) / (2.0 * sqrt(6.0));
            if (fabs(xdb) < 1e-30) xdb = 0.0;
            RSL(2,  j) = xdb;  RSL(2, -j) = xdb;

            TSL(1,  j) = 0.0;  TSL(1, -j) = 0.0;
            TSL(2,  j) = 0.0;  TSL(2, -j) = 0.0;
        }
        PSL(1, 0) = (double)RM(0);
        RSL(1, 0) = 0.0;

    } else if (is == 1) {
        /* Starting values for is=1 */
        for (int j = 0; j <= mu; j++) {
            double c = (double)RM(j);
            double x = 1.0 - c*c;
            PSL(0,  j) = 0.0;  PSL(0, -j) = 0.0;
            double sq = sqrt(x * 0.5);
            PSL(1, -j) = sq;   PSL(1,  j) = sq;
            PSL(2,  j) = c * PSL(1, j) * rac3;
            PSL(2, -j) = -PSL(2, j);

            RSL(1,  j) = 0.0;  RSL(1, -j) = 0.0;
            double xdb = -c * sqrt(x) * 0.5;
            if (fabs(xdb) < 1e-30) xdb = 0.0;
            RSL(2, -j) = -xdb;  RSL(2,  j) = xdb;  /* odd in j for is=1 */

            TSL(1,  j) = 0.0;   TSL(1, -j) = 0.0;
            xdb = -sqrt(x) * 0.5;
            if (fabs(xdb) < 1e-30) xdb = 0.0;
            TSL(2,  j) = xdb;   TSL(2, -j) = xdb;  /* even in j for is=1 */
        }
        PSL(2, 0) = -PSL(2, 0);
        RSL(2, 0) = -RSL(2, 0);
        RSL(1, 0) = 0.0;
        TSL(1, 0) = 0.0;

    } else {
        /* is >= 2: bootstrap from l = is−1 = 0, then l = is */
        double a = 1.0;
        for (int i = 1; i <= is; i++) {
            double xi = (double)i;
            a *= sqrt((xi + (double)is) / xi) * 0.5;
        }
        double b = a * sqrt((double)is / ((double)is + 1.0))
                     * sqrt(((double)is - 1.0) / ((double)is + 2.0));

        for (int j = 0; j <= mu; j++) {
            double c  = (double)RM(j);
            double xx = 1.0 - c*c;

            PSL(is-1, j) = 0.0;
            RSL(is-1, j) = 0.0;
            TSL(is-1, j) = 0.0;

            double xdb = a * pow(xx, (double)is * 0.5);
            if (fabs(xdb) < 1e-30) xdb = 0.0;
            PSL(is, -j) = xdb;  PSL(is, j) = xdb;

            xdb = b * (1.0 + c*c) * pow(xx, (double)is * 0.5 - 1.0);
            if (fabs(xdb) < 1e-30) xdb = 0.0;
            RSL(is, -j) = xdb;  RSL(is,  j) = xdb;  /* even in j */

            xdb = 2.0 * b * c * pow(xx, (double)is * 0.5 - 1.0);
            if (fabs(xdb) < 1e-30) xdb = 0.0;
            TSL(is, -j) = -xdb;  TSL(is, j) = xdb;  /* odd in j */
        }
    }

    /* ── Recurrence for l > is ── */
    int k_start = (is > 2) ? is : 2;
    if (k_start < ip1) {
        int ig = (is == 1) ? 1 : -1;   /* psl,rsl parity; tsl has -ig */
        for (int l = k_start; l <= ip1 - 1; l++) {
            int lp = l + 1, lm = l - 1;
            double lf = (double)l;
            /* psl recurrence (identical to kernel.c) */
            double aa = (2.0*lf + 1.0)
                      / sqrt((lf + (double)is + 1.0) * (lf - (double)is + 1.0));
            double bb = sqrt((lf + (double)is) * (lf - (double)is))
                      / (2.0*lf + 1.0);
            /* rsl/tsl recurrence (KERNELPOL.f d, e, f) */
            double dd = (lf + 1.0) * (2.0*lf + 1.0)
                      / sqrt((lf + 3.0) * (lf - 1.0)
                             * (lf + (double)is + 1.0)
                             * (lf - (double)is + 1.0));
            double ee = sqrt((lf + 2.0) * (lf - 2.0)
                             * (lf + (double)is)
                             * (lf - (double)is))
                      / (lf * (2.0*lf + 1.0));
            double ff = 2.0 * (double)is / (lf * (lf + 1.0));

            for (int j = 0; j <= mu; j++) {
                double c = (double)RM(j);

                double xdb = aa * (c * PSL(l, j) - bb * PSL(lm, j));
                if (fabs(xdb) < 1e-30) xdb = 0.0;
                PSL(lp, j) = xdb;
                if (j != 0) PSL(lp, -j) = (double)ig * xdb;

                xdb = dd * (c * RSL(l, j) - ff * TSL(l, j) - ee * RSL(lm, j));
                if (fabs(xdb) < 1e-30) xdb = 0.0;
                RSL(lp, j) = xdb;
                if (j != 0) RSL(lp, -j) = (double)ig * xdb;

                xdb = dd * (c * TSL(l, j) - ff * RSL(l, j) - ee * TSL(lm, j));
                if (fabs(xdb) < 1e-30) xdb = 0.0;
                TSL(lp, j) = xdb;
                if (j != 0) TSL(lp, -j) = -(double)ig * xdb;  /* tsl has opposite sign */
            }
            ig = -ig;
        }
    }

    /* ── Set xpl, xrl, xtl from the l=2 rows ── */
    for (int j = -mu; j <= mu; j++) {
        XPL(j) = PSL(2, j);
        XRL(j) = RSL(2, j);
        XTL(j) = TSL(2, j);
    }

    /* ── Compute kernel matrices ─────────────────────────────────────── */
    /* Simplified aerosol model: gammal = zetal = 0, alphal = betal.
     * Therefore:
     *   bp[j][k]  = Σ_l psl[l,j]·psl[l,k]·betal[l]   (same as scalar kernel)
     *   arr[j][k] = Σ_l rsl[l,j]·rsl[l,k]·betal[l]
     *   att[j][k] = Σ_l tsl[l,j]·tsl[l,k]·betal[l]
     *   art[j][k] = Σ_l tsl[l,j]·rsl[l,k]·betal[l]
     *   gr = gt = 0 (not stored; handled by caller via gamma2) */
    int ij = ip1;
    for (int j = 0; j <= mu; j++) {
        for (int kk = -mu; kk <= mu; kk++) {
            double sbp = 0.0, sarr = 0.0, satt = 0.0, sart = 0.0;
            if (is <= ij) {
                for (int l = is; l <= ij; l++) {
                    double bl  = (double)ctx->polar.betal[l];
                    double pjl = PSL(l, j),  pkl = PSL(l, kk);
                    double rjl = RSL(l, j),  rkl = RSL(l, kk);
                    double tjl = TSL(l, j),  tkl = TSL(l, kk);
                    sbp  += pjl * pkl * bl;
                    sarr += rjl * rkl * bl;
                    satt += tjl * tkl * bl;
                    sart += tjl * rkl * bl;
                }
            }
            if (fabs(sbp)  < 1e-30) sbp  = 0.0;
            if (fabs(sarr) < 1e-30) sarr = 0.0;
            if (fabs(satt) < 1e-30) satt = 0.0;
            if (fabs(sart) < 1e-30) sart = 0.0;
            BP (j, kk) = sbp;
            ARR(j, kk) = sarr;
            ATT(j, kk) = satt;
            ART(j, kk) = sart;
        }
    }

    #undef RM
    #undef PSL
    #undef RSL
    #undef TSL
    #undef XPL
    #undef XRL
    #undef XTL
    #undef BP
    #undef ARR
    #undef ART
    #undef ATT
}
