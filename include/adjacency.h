/* Adjacency effect correction following Vermote et al. (1997).
 *
 * In heterogeneous scenes the diffuse transmittance carries signal from
 * neighbouring pixels.  The correction replaces the uniform-surface
 * assumption by computing an environmental (spatially-averaged) reflectance
 * and subtracting its excess contribution.
 *
 *   T_diff = clip(T_scat − T_dir, 0, T_scat)
 *   r_env  = box_filter(r_boa, filter_half)
 *   r_boa  += T_diff × s_alb × (r_boa − r_env) / (1 − s_alb × r_env)
 */
#pragma once

/* Compute the environmental reflectance map via a separable box filter.
 * r_env (output) must NOT alias r_boa.
 * filter_half: half-width in pixels (total window = 2·half+1 per side). */
void adjacency_r_env(const float *r_boa, float *r_env,
                     int nrows, int ncols, int filter_half);

/* Beer-Lambert two-way direct transmittance (no multiple scattering).
 *   tau   = tau_Rayleigh(wl_um, pressure) + tau_aerosol(wl_um, aod550)
 *   T_dir = exp(−tau/cos_sza) × exp(−tau/cos_vza)
 * wl_um:    wavelength in µm
 * aod550:   AOD at 550 nm
 * pressure: atmospheric pressure in hPa
 * sza_deg, vza_deg: solar and view zenith angles in degrees */
float adjacency_T_dir(float wl_um, float aod550, float pressure,
                      float sza_deg, float vza_deg);

/* Apply Vermote 1997 adjacency correction in-place to r_boa [nrows×ncols].
 * psf_radius_km: environmental PSF radius in km (>0).
 * pixel_size_m:  pixel size in metres (>0).
 * T_scat:   two-way total scattering transmittance at this wavelength.
 * s_alb:    spherical albedo at this wavelength.
 * aod550, pressure, sza_deg, vza_deg: for T_dir computation. */
void adjacency_correct_band(float *r_boa, int nrows, int ncols,
                             float psf_radius_km, float pixel_size_m,
                             float T_scat, float s_alb,
                             float wl_um, float aod550, float pressure,
                             float sza_deg, float vza_deg);
