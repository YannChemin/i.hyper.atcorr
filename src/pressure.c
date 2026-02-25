/* Surface pressure / altitude adjustment — ported from 6SV2.1 PRESSURE.f.
 *
 * Adjusts the standard atmosphere profile stored in ctx->atm so that its
 * surface (bottom) level corresponds to the given pressure or altitude.
 * The lower layers are interpolated; the upper layers are preserved as-is.
 *
 * Convention (matching Fortran):
 *   sp > 0  : surface pressure in hPa   (most common case for elevated terrain)
 *   sp < 0  : surface altitude in km    (negative sign triggers alt-mode)
 *   sp == 0 : no-op (standard atmosphere unchanged)
 */
#include "../include/sixs_ctx.h"
#include <math.h>
#include <string.h>

/* Compute integrated H2O and O3 columns after profile adjustment.
 * Returns uw (g/cm²) and uo3 (atm-cm × 1000 Dobson). */
static void compute_columns(const SixsCtx *ctx, float *uw, float *uo3)
{
    const float g    = 98.1f;
    const float air  = 0.028964f / 0.0224f;
    const float ro3  = 0.048f / 0.0224f;

    float rmwh[NATM], rmo3[NATM];
    for (int k = 0; k < NATM - 1; k++) {
        float roair = air * 273.16f * ctx->atm.p[k]
                    / (1013.25f * ctx->atm.t[k]);
        rmwh[k] = ctx->atm.wh[k] / (roair * 1000.0f);
        rmo3[k] = ctx->atm.wo[k] / (roair * 1000.0f);
    }

    float uw_  = 0.0f, uo3_ = 0.0f;
    for (int k = 1; k < NATM - 1; k++) {
        float ds = (ctx->atm.p[k-1] - ctx->atm.p[k]) / ctx->atm.p[0];
        uw_  += 0.5f * (rmwh[k] + rmwh[k-1]) * ds;
        uo3_ += 0.5f * (rmo3[k] + rmo3[k-1]) * ds;
    }
    uw_  = uw_  * ctx->atm.p[0] * 100.0f / g;
    uo3_ = uo3_ * ctx->atm.p[0] * 100.0f / g;
    uo3_ = 1000.0f * uo3_ / ro3;

    *uw  = uw_;
    *uo3 = uo3_;
}

/* Adjust atmosphere profile to surface pressure sp (hPa, positive)
 * or surface altitude xps2 (km, negative sign → |sp|).
 * Modifies ctx->atm in place. */
void sixs_pressure(SixsCtx *ctx, float sp)
{
    if (sp == 0.0f) return;

    int isup, iinf;
    float ps, xalt;

    if (sp < 0.0f) {
        /* ── Altitude mode: sp = -altitude_km ──────────────────────────── */
        float xps2 = -sp;
        if (xps2 >= 100.0f) xps2 = 99.99f;

        /* Find bracketing levels by altitude */
        int i = 0;
        while (i < NATM - 1 && ctx->atm.z[i] <= xps2) i++;
        isup = i;
        iinf = i - 1;

        /* Log-linear pressure interpolation */
        float xa = (ctx->atm.z[isup] - ctx->atm.z[iinf])
                 / logf(ctx->atm.p[isup] / ctx->atm.p[iinf]);
        float xb = ctx->atm.z[isup] - xa * logf(ctx->atm.p[isup]);
        ps   = expf((xps2 - xb) / xa);
        xalt = xps2;
    } else {
        /* ── Pressure mode: sp = surface_pressure_hPa ────────────────── */
        if (sp >= 1013.0f) {
            /* Surface pressure higher than standard — shift bottom layers */
            float dps = sp - ctx->atm.p[0];
            for (int i = 0; i < 9 && ctx->atm.p[i] > dps; i++)
                ctx->atm.p[i] += dps;
            return;
        }
        ps = sp;

        /* Find first level where p < ps (pressure decreases upward) */
        int i = 0;
        while (i < NATM - 1 && ctx->atm.p[i] >= ps) i++;
        isup = i;
        iinf = i - 1;

        /* Log-linear altitude interpolation */
        float xa = (ctx->atm.z[isup] - ctx->atm.z[iinf])
                 / logf(ctx->atm.p[isup] / ctx->atm.p[iinf]);
        float xb = ctx->atm.z[isup] - xa * logf(ctx->atm.p[isup]);
        xalt = logf(ps) * xa + xb;
    }

    /* Linearly interpolate T, wh, wo at xalt between iinf and isup */
    float dz  = ctx->atm.z[isup] - ctx->atm.z[iinf];
    float t_frac = (dz > 0.0f) ? (xalt - ctx->atm.z[iinf]) / dz : 0.0f;

    float xtemp = ctx->atm.t[iinf]  + t_frac * (ctx->atm.t[isup]  - ctx->atm.t[iinf]);
    float xwh   = ctx->atm.wh[iinf] + t_frac * (ctx->atm.wh[isup] - ctx->atm.wh[iinf]);
    float xwo   = ctx->atm.wo[iinf] + t_frac * (ctx->atm.wo[isup] - ctx->atm.wo[iinf]);

    /* Rebuild profile: new bottom = (xalt, ps, xtemp, xwh, xwo)
     * then copy layers from iinf onward into positions 1, 2, ...
     * Fortran: do i=2, 33-iinf+1  → 0-based: i=1 to 32-iinf */
    ctx->atm.z[0]  = xalt;
    ctx->atm.p[0]  = ps;
    ctx->atm.t[0]  = xtemp;
    ctx->atm.wh[0] = xwh;
    ctx->atm.wo[0] = xwo;

    int n_copy = 33 - iinf;   /* number of levels to copy */
    for (int i = 1; i <= n_copy; i++) {
        ctx->atm.z[i]  = ctx->atm.z[i + iinf];
        ctx->atm.p[i]  = ctx->atm.p[i + iinf];
        ctx->atm.t[i]  = ctx->atm.t[i + iinf];
        ctx->atm.wh[i] = ctx->atm.wh[i + iinf];
        ctx->atm.wo[i] = ctx->atm.wo[i + iinf];
    }

    /* Fill remaining levels with linear extrapolation to TOA */
    int l = n_copy;   /* last filled index (0-based) */
    for (int i = l + 1; i < NATM; i++) {
        float frac = (float)(i - l) / (float)(NATM - 1 - l);
        ctx->atm.z[i]  = ctx->atm.z[l]  + (ctx->atm.z[NATM-1]  - ctx->atm.z[l])  * frac;
        ctx->atm.p[i]  = ctx->atm.p[l]  + (ctx->atm.p[NATM-1]  - ctx->atm.p[l])  * frac;
        ctx->atm.t[i]  = ctx->atm.t[l]  + (ctx->atm.t[NATM-1]  - ctx->atm.t[l])  * frac;
        ctx->atm.wh[i] = ctx->atm.wh[l] + (ctx->atm.wh[NATM-1] - ctx->atm.wh[l]) * frac;
        ctx->atm.wo[i] = ctx->atm.wo[l] + (ctx->atm.wo[NATM-1] - ctx->atm.wo[l]) * frac;
    }
}

/* sixs_pressure_columns: adjust profile and return integrated column amounts.
 * Thin wrapper used when callers need uw/uo3 (e.g. gas_transmittance setup). */
void sixs_pressure_columns(SixsCtx *ctx, float sp, float *uw, float *uo3)
{
    sixs_pressure(ctx, sp);
    compute_columns(ctx, uw, uo3);
}
