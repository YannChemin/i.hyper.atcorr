/* i.hyper.atcorr — image-based atmospheric state retrievals.
 *
 * Pure computation: no GRASS dependencies; compiles into both the GRASS
 * module and libatcorr.so without modification.
 * Uses sixs_E0() and sixs_earth_sun_dist2() from solar_table.c. */

#include "../include/retrieve.h"
#include "../include/atcorr.h"   /* sixs_E0, sixs_earth_sun_dist2 */

#define _USE_MATH_DEFINES
#include <math.h>
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
