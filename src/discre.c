/* Altitude-OD level finder — ported from 6SV2.1 DISCRE.f */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include "../include/sixs_ctx.h"

/* Find altitude z such that ta*exp(-z/ha) + tr*exp(-z/hr) = ti.
 * Uses bisection between ppp1 and ppp2.
 * Returns altitude in zx. Sets ctx->err.ier on error. */
void sixs_discre(SixsCtx *ctx,
                  double ta, double ha, double tr, double hr,
                  int it, int nt, double yy, double dd,
                  double ppp2, double ppp1, double *zx)
{
    if (ha >= 7.0) {
        fprintf(stderr, "atcorr: check aerosol measurements or plane altitude\n");
        ctx->err.ier = true;
        return;
    }
    double dt;
    if (it == 0) dt = 1.0e-17;
    else         dt = 2.0 * (ta + tr - yy) / (nt - it + 1.0);

again:
    dt = dt / 2.0;
    double ti = yy + dt;
    double y1 = ppp2, y3 = ppp1;

bisect:
    {
        double y2 = (y1 + y3) * 0.5;
        double xx = -y2 / ha;
        double x2;
        if (xx < -18.0) x2 = tr * exp(-y2 / hr);
        else             x2 = ta * exp(xx) + tr * exp(-y2 / hr);
        double xd = fabs(ti - x2);
        if (xd < 0.00001) {
            *zx = y2;
            double delta = 1.0 / (1.0 + ta * hr / tr / ha * exp((y2 - ppp1) * (1.0 / hr - 1.0 / ha)));
            double ecart = (dd != 0.0) ? fabs((dd - delta) / dd) : 0.0;
            if (ecart > 0.95 && it != 0) goto again;
            return;
        }
        if (ti - x2 < 0.0) y3 = y2;
        else                y1 = y2;
        goto bisect;
    }
}
