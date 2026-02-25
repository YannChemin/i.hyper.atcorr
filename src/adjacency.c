/* Adjacency effect correction — Vermote et al. (1997).
 * See include/adjacency.h for API documentation. */

#include "adjacency.h"
#include "spatial.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Environmental reflectance ──────────────────────────────────────────────── */

void adjacency_r_env(const float *r_boa, float *r_env,
                     int nrows, int ncols, int filter_half)
{
    spatial_box_filter(r_boa, r_env, nrows, ncols, filter_half);
}

/* ── Beer-Lambert direct transmittance ──────────────────────────────────────── */

float adjacency_T_dir(float wl_um, float aod550, float pressure,
                      float sza_deg, float vza_deg)
{
    /* Rayleigh optical depth (Hansen & Travis 1974) */
    float wl2   = wl_um * wl_um;
    float wl4   = wl2 * wl2;
    float tau_r = 0.008569f / wl4 * (1.0f + 0.0113f / wl2)
                  * (pressure / 1013.25f);

    /* Aerosol optical depth (power-law with alpha = 1.3) */
    float tau_a = aod550 * powf(wl_um / 0.55f, -1.3f);

    float tau = tau_r + tau_a;
    float us  = cosf(sza_deg * (float)(M_PI / 180.0));
    float uv  = cosf(vza_deg * (float)(M_PI / 180.0));

    return expf(-tau / us) * expf(-tau / uv);
}

/* ── In-place adjacency correction ─────────────────────────────────────────── */

void adjacency_correct_band(float *r_boa, int nrows, int ncols,
                             float psf_radius_km, float pixel_size_m,
                             float T_scat, float s_alb,
                             float wl_um, float aod550, float pressure,
                             float sza_deg, float vza_deg)
{
    int npix = nrows * ncols;

    /* Filter half-width: PSF radius in pixels (minimum 1) */
    int filter_half = (int)(psf_radius_km * 1000.0f / pixel_size_m + 0.5f);
    if (filter_half < 1) filter_half = 1;

    float *r_env = malloc((size_t)npix * sizeof(float));
    if (!r_env) return;

    adjacency_r_env(r_boa, r_env, nrows, ncols, filter_half);

    float T_dir  = adjacency_T_dir(wl_um, aod550, pressure, sza_deg, vza_deg);
    float T_diff = T_scat - T_dir;
    if (T_diff < 0.0f) T_diff = 0.0f;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < npix; i++) {
        float r  = r_boa[i];
        float re = r_env[i];
        if (!isfinite(r) || !isfinite(re)) continue;
        float denom = 1.0f - s_alb * re;
        if (fabsf(denom) < 1e-10f) continue;
        float corr = T_diff * s_alb * (r - re) / denom;
        r_boa[i] = r + corr;
    }

    free(r_env);
}
