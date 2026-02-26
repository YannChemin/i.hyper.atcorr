/* i.hyper.atcorr — image-based atmospheric state retrievals.
 *
 * All functions are pure computation (no GRASS dependencies).
 * The caller is responsible for allocating output buffers and handling
 * GRASS I/O (band loading, raster reading).
 *
 * References:
 *   H2O:  Kaufman & Gao (1992), IEEE TGRS 30:871-884
 *   AOD:  Kaufman et al. (1997), IEEE TGRS 35:1286-1298 (MODIS dark-target)
 *   O3:   Chappuis band depth; cross-section from Bogumil et al. (2003)
 *   Pres: ICAO International Standard Atmosphere (1993)
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* ─── H2O column from 940 nm band depth ───────────────────────────────────── *
 *
 * Continuum interpolation method (Kaufman & Gao 1992):
 *   L_cont = linear interp(L_865, L_1040) evaluated at 940 nm
 *   D      = max(0, 1 − L_940 / L_cont)
 *   WVC    = D / (K_940 × m)   where K_940 = 0.036 cm²/g, m = 1/μs + 1/μv
 *
 * L_865, L_940, L_1040: per-pixel TOA radiance [npix] at the bands closest
 *   to 865, 940, 1040 nm respectively.  NaN or ≤ 0 → default 2.0 g/cm².
 * out_wvc: pre-allocated float[npix], filled with WVC in g/cm² ∈ [0.1, 8.0].
 */
void retrieve_h2o_940(const float *L_865,  const float *L_940,
                       const float *L_1040, int npix,
                       float sza_deg, float vza_deg,
                       float *out_wvc);

/* ─── AOD from MODIS Dark Dense Vegetation (DDV) ───────────────────────────── *
 *
 * MODIS dark-target (Kaufman et al. 1997):
 *   DDV mask:   0.01 < ρ_2130 < 0.25  AND  NDVI(860,660) > 0.1
 *   Surface:    ρ_surf_470 = 0.25 ρ_2130,  ρ_surf_660 = 0.50 ρ_2130
 *   Path refl:  ρ_path = max(0, ρ_toa − ρ_surf)
 *   Inversion:  τ = ρ_path × 4μs / (ω₀ P_HG)   [nadir view assumed]
 *     g = 0.65, ω₀ = 0.89; P_HG evaluated at cos(Θ) = −μs
 *   Ångström:   τ_550 = τ_470 × (550/470)^(−α)
 *     where α = −ln(τ_470/τ_660) / ln(470/660)
 *   Non-DDV pixels filled with scene-mean AOD.
 *
 * L_470 … L_2130: per-pixel TOA radiance [npix] at ≈470/660/860/2130 nm.
 * doy: day of year (1–365); sza_deg: solar zenith angle.
 * out_aod: pre-allocated float[npix], filled with AOD at 550 nm.
 * Returns scene-mean AOD (0.15 if no DDV pixels found).
 */
float retrieve_aod_ddv(const float *L_470,  const float *L_660,
                        const float *L_860,  const float *L_2130,
                        int npix, int doy, float sza_deg,
                        float *out_aod);

/* ─── O3 column from Chappuis band depth (~600 nm) ─────────────────────────── *
 *
 * Continuum interpolation between 540 nm and 680 nm; feature at 600 nm:
 *   L_cont = linear interp(L_540, L_680) at 600 nm
 *   D      = max(0, 1 − L_600 / L_cont)
 *   O3     = D / (σ_600 × m)   σ_600 ≈ 1.0×10⁻⁴ DU⁻¹,  m = 1/μs + 1/μv
 *
 * Returns scene-mean O3 in Dobson units ∈ [50, 800]; fallback 300 DU.
 */
float retrieve_o3_chappuis(const float *L_540, const float *L_600,
                            const float *L_680, int npix,
                            float sza_deg, float vza_deg);

/* ─── ISA surface pressure from mean terrain elevation ──────────────────────── *
 *
 * International Standard Atmosphere barometric formula:
 *   P [hPa] = 1013.25 × (1 − 2.2558×10⁻⁵ × elev_m)^5.2559
 *
 * Valid for elev_m ∈ [0, 11 000] m; clamped at boundaries.
 */
float retrieve_pressure_isa(float elev_m);

#ifdef __cplusplus
}
#endif
