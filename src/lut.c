/* LUT generation — computes R_atm, T_down, T_up, s_alb arrays.
 * Calls sixs_discom() per AOD value (H2O only affects gas absorption,
 * not scattering, so DISCOM only needs to run n_aod times). */
#include "../include/sixs_ctx.h"
#include "../include/atcorr.h"
#include "interp.h"
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <string.h>
#include <stdlib.h>

/* Forward declarations for functions implemented elsewhere */
void sixs_init_atmosphere(SixsCtx *ctx, int atmo_model);
void sixs_aerosol_init(SixsCtx *ctx, int iaer, float taer55, float xmud);
void sixs_discom(SixsCtx *ctx, int idatmp, int iaer,
                  float xmus, float xmuv, float phi,
                  float taer55, float taer55p, float palt, float phirad,
                  int nt, int mu, int np, float ftray, int ipol);
float sixs_gas_transmittance(SixsCtx *ctx, float wl, float xmus, float xmuv,
                               float uw, float uo3);

/* ------------------------------------------------------------------ *
 * atcorr_compute_lut: fill LutArrays for the given LutConfig.
 *
 * LUT layout (C order, last index varies fastest):
 *   R_atm [n_aod][n_h2o][n_wl]
 *   T_down[n_aod][n_h2o][n_wl]
 *   T_up  [n_aod][n_h2o][n_wl]
 *   s_alb [n_aod][n_h2o][n_wl]  (s_alb independent of H2O, but stored for API)
 *
 * Returns 0 on success, -1 on allocation failure.
 * ------------------------------------------------------------------ */
int atcorr_compute_lut(const LutConfig *cfg, LutArrays *out)
{
    int n_wl  = cfg->n_wl;
    int n_aod = cfg->n_aod;
    int n_h2o = cfg->n_h2o;
    size_t n  = (size_t)n_aod * n_h2o * n_wl;

    if (!out->R_atm || !out->T_down || !out->T_up || !out->s_alb)
        return -1;

    /* Initialize output arrays */
    memset(out->R_atm,  0, n * sizeof(float));
    memset(out->s_alb,  0, n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        out->T_down[i] = 1.0f;
        out->T_up[i]   = 1.0f;
    }

    /* Geometry */
    float xmus   = cosf(cfg->sza * (float)M_PI / 180.0f);
    float xmuv   = cosf(cfg->vza * (float)M_PI / 180.0f);
    float phirad = cfg->raa * (float)M_PI / 180.0f;
    float phi    = cfg->raa;
    float xmud   = -xmus * xmuv -
                   sqrtf(1.0f - xmus*xmus) * sqrtf(1.0f - xmuv*xmuv) *
                   cosf(phirad);   /* cos(scattering angle) */

    /* 6SV internal parameters */
    int nt    = NT_P;     /* 30 atmospheric layers */
    int mu    = MU_P;     /* 25 Gauss points per hemisphere */
    int np    = 1;        /* one azimuth plane */
    int ipol  = 0;        /* scalar */
    int idatmp = (cfg->altitude_km > 900.0f) ? 99 :   /* satellite */
                 (cfg->altitude_km > 0.0f)   ? 4  : 0; /* plane or ground */
    float palt  = (cfg->altitude_km > 900.0f) ? 1000.0f : cfg->altitude_km;
    float ftray = 0.0f;   /* fraction above plane (0 = ground sensor) */

    /* ===== Outer loop: AOD ===== */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
    for (int ia = 0; ia < n_aod; ia++) {
        /* Each thread needs its own context for thread safety */
        SixsCtx *ctx = (SixsCtx*)calloc(1, sizeof(SixsCtx));
        if (!ctx) continue;

        float aod = cfg->aod[ia];

        /* Initialize atmosphere and aerosol */
        ctx->quad.nquad  = NQ_P;    /* 83 Gauss points */
        ctx->multi.igmax = 20;
        ctx->err.ier     = false;
        sixs_init_atmosphere(ctx, cfg->atmo_model);
        sixs_aerosol_init(ctx, cfg->aerosol_model, aod, xmud);

        /* Call DISCOM once per AOD: fills ctx->disc at 20 reference wavelengths */
        float taer55p = aod;   /* satellite: no aerosol above */
        if (cfg->altitude_km <= 900.0f) taer55p = 0.0f;
        sixs_discom(ctx, idatmp, cfg->aerosol_model,
                    xmus, xmuv, phi, aod, taer55p, palt, phirad,
                    nt, mu, np, ftray, ipol);

        if (ctx->err.ier) { free(ctx); continue; }

        /* ===== Inner loop: H2O ===== */
        for (int ih = 0; ih < n_h2o; ih++) {
            float h2o = cfg->h2o[ih];

            /* ===== Innermost loop: wavelength ===== */
            for (int iw = 0; iw < n_wl; iw++) {
                float wl = cfg->wl[iw];
                size_t idx = ((size_t)ia * n_h2o + ih) * n_wl + iw;

                /* Scattering quantities (no H2O dependence) via interpolation */
                float roatm, T_down_sca, T_up_sca, s_alb;
                sixs_interp(ctx, cfg->aerosol_model, wl,
                             aod, taer55p,
                             &roatm, &T_down_sca, &T_up_sca, &s_alb,
                             NULL, NULL);

                /* Gas transmittance (H2O-dependent): separate solar/view paths */
                float T_gas_down = sixs_gas_transmittance(ctx, wl, xmus, xmuv,
                                                           h2o, cfg->ozone_du);
                float T_gas_up   = sixs_gas_transmittance(ctx, wl, xmuv, xmuv,
                                                           h2o, cfg->ozone_du);

                /* Combined transmittances */
                out->R_atm [idx] = roatm;
                out->T_down[idx] = T_down_sca * T_gas_down;
                out->T_up  [idx] = T_up_sca   * T_gas_up;
                out->s_alb [idx] = s_alb;
            }
        }

        free(ctx);
    }

    return 0;
}
