/* Gas absorption transmittances — ported from 6SV2.1 ABSTRA.f */
#include "../include/sixs_ctx.h"
#include "../include/gas_tables.h"
#include <math.h>
#include <string.h>

/* Band boundaries in cm^-1 (ivli in ABSTRA.f) */
static const int ivli[6] = {2500, 5060, 7620, 10180, 12740, 15300};

/* Get the 8 absorption coefficients for a given gas, band, interval.
 * gas_id: 1=H2O, 2=CO2, 3=O2, 5=N2O, 6=CH4, 7=CO
 * band: 1-6, inu: 1-256 */
static void get_gas_coef(int gas_id, int band, int inu, float a[8]) {
    /* Map gas_id to table index (0-based) */
    const float (*tbl)[256][8] = NULL;
    switch (gas_id) {
        case 1: tbl = gas_acr_wava; break;
        case 2: tbl = gas_acr_dica; break;
        case 3: tbl = gas_acr_oxyg; break;
        case 5: tbl = gas_acr_niox; break;
        case 6: tbl = gas_acr_meth; break;
        case 7: tbl = gas_acr_moca; break;
        default: memset(a, 0, 8 * sizeof(float)); return;
    }
    int b = band - 1;
    int i = inu - 1;
    if (b < 0 || b >= 6 || i < 0 || i >= 256) {
        memset(a, 0, 8 * sizeof(float)); return;
    }
    for (int k = 0; k < 8; k++) a[k] = tbl[b][i][k];
}

/* Compute gas transmittance for a single gas species.
 * Uses the 6SV Curtis-Godson approximation.
 * Returns transmittance (0-1). */
static float compute_gas_trans(const SixsCtx *ctx, const float a[8],
                                double ud, double upd, int gas_id)
{
    if (a[0] == 0.0f && a[1] == 0.0f) return 1.0f;
    double udt  = (ud  == 0.0 && upd == 0.0) ? 1.0 : ud;
    double updt = (ud  == 0.0 && upd == 0.0) ? 1.0 : upd;
    double atest = (a[1] == 0.0f && a[0] == 0.0f) ? 1.0 : (double)a[1];
    double y;
    if (gas_id == 1) {
        /* H2O: special formula */
        y = -(double)a[0] * ud / sqrt(1.0 + ((double)a[0] / atest) * (ud * ud / updt));
    } else {
        double tn = (double)a[1] * upd / (2.0 * udt);
        double tt = 1.0 + 4.0 * ((double)a[0] / atest) * (ud * ud / updt);
        y = -tn * (sqrt(tt) - 1.0);
    }
    if (y < -86.0) return 0.0f;
    return (float)exp(y);
}

/* Compute gas column amounts (u, up) for a given gas species and direction.
 * Returns u (total column) and up (pressure-weighted column). */
static void compute_column(const SixsCtx *ctx, int gas_id, const float a[8],
                             float xmus, double *u_out, double *up_out)
{
    const float *z  = ctx->atm.z;
    const float *p  = ctx->atm.p;
    const float *t  = ctx->atm.t;
    const float *wh = ctx->atm.wh;
    const float *wo = ctx->atm.wo;

    const float t0    = 250.0f;
    const float p0    = 1013.25f;
    const float g     = 98.1f;
    const float air   = 0.028964f / 0.0224f;
    const float roco2 = 0.044f   / 0.0224f;
    const float rmo2  = 0.032f   / 0.0224f;
    const float rmo3  = 0.048f   / 0.0224f;
    const float rmn2o = 0.044f   / 0.0224f;
    const float rmch4 = 0.016f   / 0.0224f;
    const float rmco  = 0.028f   / 0.0224f;

    double uu = 0.0, u = 0.0, up = 0.0;
    for (int k = 1; k < NATM - 1; k++) {
        float tp = (t[k] + t[k+1]) / 2.0f;
        float roair = air * 273.16f * p[k] / (1013.25f * t[k]);
        float te  = tp - t0;
        float te2 = te * te;
        float phi = expf((float)(a[2]) * te + (float)(a[3]) * te2);
        float psi = expf((float)(a[4]) * te + (float)(a[5]) * te2);
        float rm_k;
        switch (gas_id) {
            case 1: rm_k = wh[k] / (roair * 1000.0f); break;
            case 2: rm_k = 3.3e-4f * roco2 / air; break;
            case 3: rm_k = 0.20947f * rmo2 / air; break;
            case 4: rm_k = wo[k] / (roair * 1000.0f); break;
            case 5: rm_k = 310.0e-9f * rmn2o / air; break;
            case 6: rm_k = 1.72e-6f * rmch4 / air; break;
            case 7: rm_k = 1.00e-9f * rmco  / air; break;
            default: rm_k = 0.0f; break;
        }
        float r2 = rm_k * phi;
        float r3 = rm_k * psi;
        float ds  = (p[k] - p[k+1]) / p[0];
        float ds2 = (p[k] * p[k] - p[k+1] * p[k+1]) / (2.0f * p[0] * p0);
        uu += rm_k * ds;
        u  += r2   * ds;
        up += r3   * ds2;
    }
    uu *= (double)p[0] * 100.0 / g;
    u  *= (double)p[0] * 100.0 / g;
    up *= (double)p[0] * 100.0 / g;

    /* Unit conversions */
    if (gas_id == 4) uu *= 1000.0 / rmo3;
    if (gas_id == 2) uu *= 1000.0 / roco2;
    if (gas_id == 5) uu *= 1000.0 / rmn2o;
    if (gas_id == 6) uu *= 1000.0 / rmch4;
    if (gas_id == 7) uu *= 1000.0 / rmco;

    *u_out  = u / xmus;
    *up_out = up / xmus;
}

/* Main gas absorption function.
 * Returns downward transmittance for all gases combined. */
float sixs_gas_transmittance(const SixsCtx *ctx, float wl, float xmus, float xmuv,
                              float uw, float uo3)
{
    if (wl <= 0.0f) return 1.0f;
    float v  = 1.0e4f / wl;
    int   iv = (int)(v / 5.0f) * 5;
    /* Band index */
    int id = ((iv - 2500) / 10) / 256 + 1;
    if (id < 1 || id > 6) return 1.0f;
    int inu = (iv - ivli[id - 1]) / 10 + 1;
    if (inu < 1 || inu > 256) return 1.0f;

    float tgas_down = 1.0f;

    /* Loop over gas species (ABSTRA loops idgaz=1..7) */
    for (int gas_id = 1; gas_id <= 7; gas_id++) {
        /* Check if this gas/band combination exists */
        int has_data = 0;
        if (gas_id == 1) has_data = 1;                         /* H2O: all bands */
        if (gas_id == 2 && id <= 3) has_data = 1;             /* CO2: bands 1-3 */
        if (gas_id == 3 && id >= 3) has_data = 1;             /* O2:  bands 3-6 */
        if (gas_id == 4) { /* O3: special treatment below */ continue; }
        if (gas_id >= 5) has_data = 1;                        /* N2O/CH4/CO: all bands */
        if (!has_data) continue;

        float a[8];
        get_gas_coef(gas_id, id, inu, a);
        if (a[0] == 0.0f) continue;

        /* H2O: scale by uw/uwus (uwus=1.424 g/cm²) */
        float scale = 1.0f;
        if (gas_id == 1 && uw > 0.0f) scale = uw / 1.424f;

        double ud, upd;
        compute_column(ctx, gas_id, a, xmus, &ud, &upd);
        ud  *= scale;
        upd *= scale;

        float t_d = compute_gas_trans(ctx, a, ud, upd, gas_id);
        tgas_down *= t_d;
    }

    /* Ozone absorption (separate table: co3[102]) */
    if (iv >= 13000 && iv <= 27400) {
        extern const float gas_co3_ozon[102];
        float xi;
        if (iv <= 23400) xi = (v - 13000.0f) / 200.0f + 1.0f;
        else             xi = (v - 27500.0f) / 500.0f + 57.0f;
        int n = (int)(xi + 1.001f);
        float xd = xi - (float)n;
        if (n >= 1 && n <= 102) {
            float ako3 = gas_co3_ozon[n - 1] + xd * (gas_co3_ozon[n - 1] - (n > 1 ? gas_co3_ozon[n - 2] : gas_co3_ozon[0]));
            /* Ozone column: uo3 in Dobson units / 1000 = atm-cm */
            float uud = (uo3 > 0.0f ? uo3 / 1000.0f : 0.344f) / xmus;
            float test = ako3 * uud;
            if (test > 86.0f) test = 86.0f;
            tgas_down *= expf(-test);
        }
    }

    return tgas_down;
}

/* Compute both downward and upward gas transmittances */
void sixs_gas_transmittances(const SixsCtx *ctx, float wl, float xmus, float xmuv,
                              float uw, float uo3,
                              float *t_down, float *t_up)
{
    *t_down = sixs_gas_transmittance(ctx, wl, xmus, xmuv, uw, uo3);
    *t_up   = sixs_gas_transmittance(ctx, wl, xmuv, xmuv, uw, uo3);
}
