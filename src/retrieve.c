/**
 * \file retrieve.c
 * \brief Image-based atmospheric state retrievals (H₂O, AOD, O₃, pressure, quality).
 *
 * Pure computation: no GRASS dependencies; compiles into both the GRASS
 * module and libatcorr.so without modification.
 * Uses sixs_E0() and sixs_earth_sun_dist2() from solar_table.c.
 *
 * \see include/retrieve.h for the public API and algorithm descriptions.
 */

#include "../include/retrieve.h"
#include "../include/atcorr.h"   /* sixs_E0, sixs_earth_sun_dist2 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── H2O 940 nm retrieval ────────────────────────────────────────────────── */

void retrieve_h2o_940(const float *L_865,  const float *L_940,
                       const float *L_1040, int npix,
                       float sza_deg, float vza_deg,
                       float *out_wvc)
{
    /* Fraction of the 865→1040 range occupied by 940-865 = 75/175 ≈ 0.4286 */
    static const float FRAC    = (0.940f - 0.865f) / (1.040f - 0.865f);
    static const float K_940   = 0.036f;   /* cm²/g; empirical at 940 nm */
    static const float WVC_DEF = 2.0f;     /* fallback for invalid pixels */
    static const float WVC_MIN = 0.1f;
    static const float WVC_MAX = 8.0f;

    float cos_sza = cosf(sza_deg * (float)(M_PI / 180.0));
    float cos_vza = cosf(vza_deg * (float)(M_PI / 180.0));
    float mu_s = (cos_sza > 0.05f) ? cos_sza : 0.05f;
    float mu_v = (cos_vza > 0.05f) ? cos_vza : 0.05f;
    float km   = K_940 * (1.0f / mu_s + 1.0f / mu_v);

    for (int i = 0; i < npix; i++) {
        float l8 = L_865[i], l9 = L_940[i], l10 = L_1040[i];
        if (!(l8 > 0.0f) || !(l9 > 0.0f) || !(l10 > 0.0f)) {
            out_wvc[i] = WVC_DEF;
            continue;
        }
        float L_cont = l8 * (1.0f - FRAC) + l10 * FRAC;
        if (L_cont <= 0.0f) { out_wvc[i] = WVC_DEF; continue; }

        float D = 1.0f - l9 / L_cont;
        if (D < 0.0f) D = 0.0f;

        float wvc = D / km;
        out_wvc[i] = (wvc < WVC_MIN) ? WVC_MIN : (wvc > WVC_MAX) ? WVC_MAX : wvc;
    }
}

/* ─── AOD DDV retrieval ────────────────────────────────────────────────────── */

float retrieve_aod_ddv(const float *L_470,  const float *L_660,
                        const float *L_860,  const float *L_2130,
                        int npix, int doy, float sza_deg,
                        float *out_aod)
{
    static const float G      = 0.65f;   /* HG asymmetry parameter */
    static const float OMEGA0 = 0.89f;   /* single-scattering albedo */
    static const float AOD_FALLBACK = 0.15f;
    static const float AOD_MAX      = 3.0f;

    float E0_470  = sixs_E0(0.470f);
    float E0_660  = sixs_E0(0.660f);
    float E0_2130 = sixs_E0(2.130f);
    float d2f     = (float)sixs_earth_sun_dist2(doy);

    float cos_sza = cosf(sza_deg * (float)(M_PI / 180.0));
    float mu_s = (cos_sza > 0.05f) ? cos_sza : 0.05f;

    /* Henyey-Greenstein phase function for nadir view (cos Θ = −μs):
     * P_HG = (1 − g²) / (1 + g² − 2g cos Θ)^1.5  with cos Θ = −μs */
    float cos_theta = -mu_s;
    float denom_hg  = 1.0f + G*G - 2.0f*G*cos_theta;
    float P_HG = (1.0f - G*G) / (denom_hg * sqrtf(denom_hg));
    if (P_HG < 1e-4f) P_HG = 1e-4f;

    /* ρ_toa = π × L × d² / (E0 × μs)  →  scale = π × d² / μs */
    float pi_d2_over_mus = (float)M_PI * d2f / mu_s;

    /* τ = ρ_path × 4 μs / (ω₀ P_HG)  with μv = 1 (nadir) */
    float tau_factor = 4.0f * mu_s / (OMEGA0 * P_HG);

    double sum_aod = 0.0;
    int    n_ddv   = 0;

    for (int i = 0; i < npix; i++) {
        float l4 = L_470[i], l6 = L_660[i], l8 = L_860[i], l21 = L_2130[i];
        if (!(l4 > 0.0f) || !(l6 > 0.0f) || !(l8 > 0.0f) || !(l21 > 0.0f)) {
            out_aod[i] = -1.0f;   /* invalid → fill with mean later */
            continue;
        }

        /* TOA reflectance at 2130 nm for DDV mask */
        float rho_2130 = pi_d2_over_mus * l21 / E0_2130;
        if (rho_2130 < 0.01f || rho_2130 > 0.25f) {
            out_aod[i] = -1.0f;
            continue;
        }

        /* NDVI from 860/660 radiance (E0 factor cancels in ratio) */
        float ndvi = (l8 - l6) / (l8 + l6 + 1e-10f);
        if (ndvi <= 0.1f) { out_aod[i] = -1.0f; continue; }

        /* TOA reflectance at 470/660 nm */
        float rho_470 = pi_d2_over_mus * l4 / E0_470;
        float rho_660 = pi_d2_over_mus * l6 / E0_660;

        /* DDV surface reflectance prediction */
        float rho_surf_470 = 0.25f * rho_2130;
        float rho_surf_660 = 0.50f * rho_2130;

        /* Path reflectance (ensure positive) */
        float rho_path_470 = rho_470 - rho_surf_470;
        float rho_path_660 = rho_660 - rho_surf_660;
        if (rho_path_470 < 1e-5f) rho_path_470 = 1e-5f;
        if (rho_path_660 < 1e-5f) rho_path_660 = 1e-5f;

        /* Single-scattering aerosol optical depth at 470/660 nm */
        float tau_470 = rho_path_470 * tau_factor;
        float tau_660 = rho_path_660 * tau_factor;

        /* Ångström exponent and scale to 550 nm */
        float alpha = 0.0f;
        if (tau_470 > 1e-4f && tau_660 > 1e-4f) {
            alpha = -logf(tau_470 / tau_660) / logf(0.470f / 0.660f);
            if (alpha < -1.0f) alpha = -1.0f;
            if (alpha >  3.0f) alpha =  3.0f;
        }
        float tau_550 = tau_470 * powf(0.550f / 0.470f, -alpha);
        if (tau_550 < 0.0f)   tau_550 = 0.0f;
        if (tau_550 > AOD_MAX) tau_550 = AOD_MAX;

        out_aod[i] = tau_550;
        sum_aod   += tau_550;
        n_ddv++;
    }

    float aod_mean = (n_ddv > 0) ? (float)(sum_aod / n_ddv) : AOD_FALLBACK;

    /* Fill invalid pixels (marked -1) with scene mean */
    for (int i = 0; i < npix; i++)
        if (out_aod[i] < 0.0f) out_aod[i] = aod_mean;

    return aod_mean;
}

/* ─── O3 Chappuis band retrieval ───────────────────────────────────────────── */

float retrieve_o3_chappuis(const float *L_540, const float *L_600,
                            const float *L_680, int npix,
                            float sza_deg, float vza_deg)
{
    /* Fraction of 540→680 range at 600 nm: (600-540)/(680-540) = 3/7 */
    static const float FRAC       = (0.600f - 0.540f) / (0.680f - 0.540f);
    /* Chappuis cross section at 600 nm; effective DU⁻¹ absorption coefficient.
     * Derived from Bogumil et al. (2003): σ ≈ 3.8e-21 cm²/mol;
     * 1 DU = 2.687e16 mol/cm²  →  σ_DU = 3.8e-21 × 2.687e16 ≈ 1.02e-4 DU⁻¹ */
    static const float SIGMA_O3   = 1.0e-4f;
    static const float O3_FALLBACK = 300.0f;
    static const float O3_MIN      = 50.0f;
    static const float O3_MAX      = 800.0f;

    float cos_sza = cosf(sza_deg * (float)(M_PI / 180.0));
    float cos_vza = cosf(vza_deg * (float)(M_PI / 180.0));
    float mu_s = (cos_sza > 0.05f) ? cos_sza : 0.05f;
    float mu_v = (cos_vza > 0.05f) ? cos_vza : 0.05f;
    float sigma_m = SIGMA_O3 * (1.0f / mu_s + 1.0f / mu_v);

    double sum_o3  = 0.0;
    int    n_valid = 0;

    for (int i = 0; i < npix; i++) {
        float l5 = L_540[i], l6 = L_600[i], l7 = L_680[i];
        if (!(l5 > 0.0f) || !(l6 > 0.0f) || !(l7 > 0.0f)) continue;

        float L_cont = l5 * (1.0f - FRAC) + l7 * FRAC;
        if (L_cont <= 0.0f || l6 >= L_cont) continue;

        float D  = 1.0f - l6 / L_cont;
        float o3 = D / sigma_m;

        if (o3 >= O3_MIN && o3 <= O3_MAX) {
            sum_o3 += o3;
            n_valid++;
        }
    }

    if (n_valid == 0) return O3_FALLBACK;

    float result = (float)(sum_o3 / n_valid);
    return (result < O3_MIN) ? O3_MIN : (result > O3_MAX) ? O3_MAX : result;
}

/* ─── ISA surface pressure ─────────────────────────────────────────────────── */

float retrieve_pressure_isa(float elev_m)
{
    /* ICAO International Standard Atmosphere (1993)
     * P = 1013.25 × (1 − 2.2558×10⁻⁵ × h)^5.2559  [hPa] */
    if (elev_m < 0.0f)       elev_m = 0.0f;
    if (elev_m > 11000.0f)   elev_m = 11000.0f;
    return 1013.25f * powf(1.0f - 2.2558e-5f * elev_m, 5.2559f);
}

/* ─── O₂-A surface pressure retrieval ─────────────────────────────────────── */

void retrieve_pressure_o2a(const float *L_740, const float *L_760,
                             const float *L_780, int npix,
                             float sza_deg, float vza_deg,
                             float *out_pressure)
{
    /* Continuum interpolation between 740 nm and 780 nm; feature at 760 nm.
     *
     * Physical model (Beer-Lambert, linearised):
     *   τ_O2(P) = K_O2 × (P / P0)
     *   D_760   = max(0, 1 − L_760 / L_cont) ≈ τ_O2 × m  (first-order)
     *   P       = P0 × D_760 / (K_O2 × m)
     *
     * K_O2 = 0.25: calibrated to ~10 nm FWHM sensors; derived from
     * Schläpfer et al. (1998): at P=P0, SZA=0 (m=2), band-depth D ≈ 0.50.
     * For a 5 nm sensor set K_O2 ≈ 0.40; for 20 nm use K_O2 ≈ 0.15.
     *
     * Reference:
     *   Schläpfer, D., Borel, C.C., Keller, J., Itten, K.I. (1998).
     *   Atmospheric pre-processing of DAIS airborne data. SPIE 3502.
     */
    static const float FRAC   = 0.5f;      /* (760−740)/(780−740) = 0.5 */
    static const float K_O2   = 0.25f;     /* effective OD per unit air mass at P0 */
    static const float P0     = 1013.25f;  /* sea-level pressure [hPa] */
    static const float P_MIN  = 200.0f;    /* ~11 km altitude */
    static const float P_MAX  = 1100.0f;
    static const float P_DEF  = P0;

    float cos_sza = cosf(sza_deg * (float)(M_PI / 180.0));
    float cos_vza = cosf(vza_deg * (float)(M_PI / 180.0));
    float mu_s = (cos_sza > 0.05f) ? cos_sza : 0.05f;
    float mu_v = (cos_vza > 0.05f) ? cos_vza : 0.05f;
    float m    = 1.0f / mu_s + 1.0f / mu_v;   /* two-way air mass */

    for (int i = 0; i < npix; i++) {
        float l74 = L_740[i], l76 = L_760[i], l78 = L_780[i];
        if (!(l74 > 0.0f) || !(l76 > 0.0f) || !(l78 > 0.0f)) {
            out_pressure[i] = P_DEF;
            continue;
        }
        float L_cont = l74 * (1.0f - FRAC) + l78 * FRAC;
        if (L_cont <= 0.0f || l76 >= L_cont) {
            out_pressure[i] = P_DEF;
            continue;
        }
        float D = 1.0f - l76 / L_cont;
        if (D < 0.0f) D = 0.0f;
        if (D > 0.95f) D = 0.95f;  /* cap: near-zero pressure is unphysical */

        float P = P0 * D / (K_O2 * m);
        out_pressure[i] = (P < P_MIN) ? P_MIN : (P > P_MAX) ? P_MAX : P;
    }
}

/* ─── Cloud / shadow / water / snow quality bitmask ───────────────────────── */

#include <stdint.h>

void retrieve_quality_mask(const float *L_blue, const float *L_red,
                            const float *L_nir,  const float *L_swir,
                            int npix, int doy, float sza_deg,
                            uint8_t *out_mask)
{
    /* Thresholds are applied to TOA reflectance (from TOA radiance).
     *
     * Bitmask:  MASK_CLOUD=0x01  MASK_SHADOW=0x02  MASK_WATER=0x04  MASK_SNOW=0x08
     *
     * Cloud:    TOA_blue > 0.25 AND NDVI < 0.2, OR TOA_nir > 0.60
     * Shadow:   TOA_blue < 0.04 AND TOA_red < 0.04 AND TOA_nir < 0.04
     * Water:    TOA_nir < 0.05 AND NDVI < 0.0
     * Snow/ice: NDSI > 0.40 AND TOA_nir > 0.10   (requires L_swir)
     *
     * References:
     *   Cloud / shadow: Braaten et al. (2015), Remote Sens. 7:15745
     *   Water:          McFeeters (1996), Int. J. Remote Sens. 17:1425
     *   Snow:           Hall et al. (1995), Remote Sens. Environ. 54:127
     */
    float E0_blue = sixs_E0(0.470f);
    float E0_red  = sixs_E0(0.660f);
    float E0_nir  = sixs_E0(0.860f);
    float E0_swir = sixs_E0(1.600f);
    float d2f     = (float)sixs_earth_sun_dist2(doy);

    float cos_sza = cosf(sza_deg * (float)(M_PI / 180.0));
    float mu_s    = (cos_sza > 0.05f) ? cos_sza : 0.05f;
    float scale   = (float)M_PI * d2f / mu_s;

    for (int i = 0; i < npix; i++) {
        out_mask[i] = 0;
        float lb = L_blue[i], lr = L_red[i], ln = L_nir[i];
        if (!(lb > 0.0f) || !(lr > 0.0f) || !(ln > 0.0f)) continue;

        float rb = scale * lb / E0_blue;
        float rr = scale * lr / E0_red;
        float rn = scale * ln / E0_nir;

        float ndvi = (rn - rr) / (rn + rr + 1e-10f);

        /* NDSI requires SWIR */
        float ndsi = -2.0f;
        if (L_swir && L_swir[i] > 0.0f) {
            float rs = scale * L_swir[i] / E0_swir;
            ndsi = (rb - rs) / (rb + rs + 1e-10f);
        }

        if ((rb > 0.25f && ndvi < 0.2f) || rn > 0.60f)
            out_mask[i] |= RETRIEVE_MASK_CLOUD;

        if (rb < 0.04f && rr < 0.04f && rn < 0.04f)
            out_mask[i] |= RETRIEVE_MASK_SHADOW;

        if (rn < 0.05f && ndvi < 0.0f)
            out_mask[i] |= RETRIEVE_MASK_WATER;

        if (ndsi > 0.40f && rn > 0.10f)
            out_mask[i] |= RETRIEVE_MASK_SNOW;
    }
}

/* ─── MAIAC-inspired patch AOD spatial regularization ─────────────────────── */

static int _float_cmp(const void *a, const void *b)
{
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

void retrieve_aod_maiac(float *aod_data, int nrows, int ncols, int patch_sz)
{
    /* Segment image into non-overlapping patch_sz × patch_sz blocks.
     * Each patch with ≥1 valid pixel uses the patch-median AOD (robust to
     * DDV outliers).  Invalid patches are filled by inverse-distance
     * weighting from the nearest valid patch centroids.
     *
     * Reference:
     *   Lyapustin, A. et al. (2011). Multiangle implementation of atmospheric
     *   correction (MAIAC) — Part 1. JGR 116, D05203.  This implementation
     *   uses only the spatial regularization concept (patch-median + IDW fill),
     *   not the full MAIAC surface model or multi-angle retrieval.
     */
    if (patch_sz < 2) patch_sz = 2;

    int np_rows  = (nrows + patch_sz - 1) / patch_sz;
    int np_cols  = (ncols + patch_sz - 1) / patch_sz;
    int np_total = np_rows * np_cols;
    int buf_max  = patch_sz * patch_sz;

    float *patch_aod   = malloc((size_t)np_total * sizeof(float));
    float *patch_row   = malloc((size_t)np_total * sizeof(float));
    float *patch_col   = malloc((size_t)np_total * sizeof(float));
    int   *patch_valid = malloc((size_t)np_total * sizeof(int));
    float *patch_buf   = malloc((size_t)buf_max   * sizeof(float));
    if (!patch_aod || !patch_row || !patch_col || !patch_valid || !patch_buf) {
        free(patch_aod); free(patch_row); free(patch_col);
        free(patch_valid); free(patch_buf);
        return;   /* silently skip on OOM */
    }

    /* ── Step 1: per-patch median ── */
    for (int pr = 0; pr < np_rows; pr++) {
        for (int pc = 0; pc < np_cols; pc++) {
            int pi = pr * np_cols + pc;
            int r0 = pr * patch_sz, r1 = r0 + patch_sz; if (r1 > nrows) r1 = nrows;
            int c0 = pc * patch_sz, c1 = c0 + patch_sz; if (c1 > ncols) c1 = ncols;

            patch_row[pi] = 0.5f * (float)(r0 + r1);
            patch_col[pi] = 0.5f * (float)(c0 + c1);

            int n = 0;
            for (int r = r0; r < r1; r++)
                for (int c = c0; c < c1; c++) {
                    float v = aod_data[r * ncols + c];
                    if (isfinite(v) && v >= 0.0f)
                        patch_buf[n++] = v;
                }

            if (n == 0) {
                patch_valid[pi] = 0;
                patch_aod[pi]   = -1.0f;
            } else {
                qsort(patch_buf, (size_t)n, sizeof(float), _float_cmp);
                patch_aod[pi]   = patch_buf[n / 2];
                patch_valid[pi] = 1;
            }
        }
    }

    /* ── Step 2: IDW fill for invalid patches ── */
    for (int pi = 0; pi < np_total; pi++) {
        if (patch_valid[pi]) continue;
        double sum_w = 0.0, sum_wa = 0.0;
        for (int pj = 0; pj < np_total; pj++) {
            if (!patch_valid[pj]) continue;
            float dr = patch_row[pi] - patch_row[pj];
            float dc = patch_col[pi] - patch_col[pj];
            double d2 = (double)(dr * dr + dc * dc);
            if (d2 < 1.0) d2 = 1.0;
            double w = 1.0 / d2;
            sum_w  += w;
            sum_wa += w * (double)patch_aod[pj];
        }
        patch_aod[pi] = (sum_w > 0.0) ? (float)(sum_wa / sum_w) : 0.1f;
    }

    /* ── Step 3: write patch medians back to all pixels in each patch ── */
    for (int pr = 0; pr < np_rows; pr++) {
        for (int pc = 0; pc < np_cols; pc++) {
            int pi = pr * np_cols + pc;
            float aod_p = patch_aod[pi];
            int r0 = pr * patch_sz, r1 = r0 + patch_sz; if (r1 > nrows) r1 = nrows;
            int c0 = pc * patch_sz, c1 = c0 + patch_sz; if (c1 > ncols) c1 = ncols;
            for (int r = r0; r < r1; r++)
                for (int c = c0; c < c1; c++)
                    aod_data[r * ncols + c] = aod_p;
        }
    }

    free(patch_aod); free(patch_row); free(patch_col);
    free(patch_valid); free(patch_buf);
}
