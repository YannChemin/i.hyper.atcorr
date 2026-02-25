/* Reflectance uncertainty propagation (#4) and model discrepancy (#6).
 *
 * Two uncertainty sources are propagated through the algebraic inversion:
 *   1. Instrument noise: NEDL → σ_ρ_toa → σ_rfl = σ_ρ_toa / (T_down × T_up)
 *   2. AOD uncertainty: perturb AOD by ±aod_sigma, take half-difference
 *
 * Model discrepancy is a per-band RT error floor (see surface_model_discrepancy).
 * It is added in quadrature to the propagated uncertainty before MAP regularisation.
 */
#pragma once

#include "../include/atcorr.h"

/* Estimate noise-equivalent delta radiance (NEDL) from the standard deviation
 * of the darkest 5 % of finite positive radiance pixels.
 * Returns 0.01 as a fallback when fewer than 10 valid pixels exist. */
float uncertainty_estimate_nedl(const float *rad, int npix);

/* Compute per-pixel reflectance uncertainty for one band.
 *
 * Outputs sigma_out[npix] = sqrt(σ_noise² + σ_aod²).
 *
 * Parameters:
 *   rad_band:  [npix] TOA radiance (may be NULL → nedl used as-is)
 *   refl_band: [npix] BOA reflectance from atcorr_invert()
 *   E0:        exo-atmospheric irradiance W/(m² µm)
 *   d2:        squared Earth-Sun distance (AU²)
 *   cos_sza:   cosine of solar zenith angle
 *   T_down/T_up/s_alb/R_atm: scene-average LUT values at this wavelength
 *   nedl:      noise-equivalent ΔL; 0 → estimate from rad_band
 *   aod_sigma: AOD perturbation half-width (default 0.04)
 *   cfg, lut:  LUT for AOD perturbation
 *   wl_um:     wavelength in µm
 *   aod_val, h2o_val: scene-average atmospheric state
 *   sigma_out: [npix] output (NaN where refl_band is NaN) */
void uncertainty_compute_band(const float *rad_band, const float *refl_band,
                               int npix,
                               float E0, float d2, float cos_sza,
                               float T_down, float T_up,
                               float s_alb,  float R_atm,
                               float nedl,   float aod_sigma,
                               const LutConfig  *cfg,
                               const LutArrays  *lut,
                               float wl_um,
                               float aod_val, float h2o_val,
                               float *sigma_out);
