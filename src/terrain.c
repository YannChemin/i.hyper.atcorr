/**
 * \file terrain.c
 * \brief Terrain illumination and transmittance correction for tilted surfaces.
 *
 * Implements the topographic corrections described in:
 *   Richter & Schläpfer (2002), Topographic correction of satellite data using
 *   a modified Minnaert approach; also consistent with Tanre et al. (1992) and
 *   the correction used in ATCOR and PARGE.
 *
 * Reference for diffuse skyview decomposition:
 *   Häberle & Richter (2015), ATCOR Technical Report.
 *
 * Physics:
 *   On a tilted surface the downward solar irradiance is split into:
 *     E_direct  = E0 × T_down_dir × cos_i       [local incidence]
 *     E_diffuse = E0 × T_down_dif × Vd          [skyview-weighted]
 *   where Vd = (1 + cos(slope)) / 2 and T_down_dif = T_down − T_down_dir.
 *
 *   The effective downward transmittance that enters the inversion is:
 *     T_down_eff = T_down_dir × (cos_i / cos_sza) + T_down_dif × Vd
 *   (divided by E0×cos_sza as in the flat-surface formula).
 */
#include "../include/terrain.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Local illumination angle ────────────────────────────────────────────── */

float cos_incidence(float sza_deg, float saa_deg,
                    float slope_deg, float aspect_deg)
{
    float sza    = sza_deg    * (float)(M_PI / 180.0);
    float saa    = saa_deg    * (float)(M_PI / 180.0);
    float slope  = slope_deg  * (float)(M_PI / 180.0);
    float aspect = aspect_deg * (float)(M_PI / 180.0);

    /* Standard topographic illumination model (Iqbal 1983) */
    return cosf(sza) * cosf(slope)
         + sinf(sza) * sinf(slope) * cosf(saa - aspect);
}

/* ── Skyview factor ───────────────────────────────────────────────────────── */

float skyview_factor(float slope_deg)
{
    float s = slope_deg * (float)(M_PI / 180.0);
    return (1.0f + cosf(s)) * 0.5f;
}

/* ── Effective downward transmittance on tilted surface ─────────────────── */

float atcorr_terrain_T_down(float T_down, float T_down_dir,
                             float cos_sza, float cos_i, float V_d)
{
    float T_dif = T_down - T_down_dir;
    if (T_dif < 0.0f) T_dif = 0.0f;

    if (cos_i <= 0.0f)
        /* Topographic shadow: only diffuse sky radiation reaches surface */
        return T_dif * V_d;

    /* Illuminated: direct component scaled by local/scene incidence ratio */
    float cos_sza_safe = (cos_sza > 1e-6f) ? cos_sza : 1e-6f;
    return T_down_dir * (cos_i / cos_sza_safe) + T_dif * V_d;
}

/* ── Per-pixel upward transmittance correction ───────────────────────────── */

float atcorr_terrain_T_up(float T_up, float cos_vza_ref, float vza_pixel_deg)
{
    if (T_up <= 0.0f || T_up >= 1.0f) return T_up;

    float cos_px = cosf(vza_pixel_deg * (float)(M_PI / 180.0));
    if (cos_px < 1e-4f) cos_px = 1e-4f;
    if (cos_vza_ref < 1e-4f) cos_vza_ref = 1e-4f;

    /* T_up ∝ exp(−τ / cos_vza)  →  T_up(px) = T_up(ref) ^ (cos_ref / cos_px) */
    return powf(T_up, cos_vza_ref / cos_px);
}
