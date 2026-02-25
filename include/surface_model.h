/* Surface reflectance prior: 3-component Gaussian mixture.
 *
 * Components: 0=vegetation, 1=soil, 2=water.
 * Reference spectra are hardcoded at 29 key wavelengths (0.40–2.50 µm) and
 * linearly interpolated to sensor wavelengths at initialisation time.
 *
 * Used for:
 *   #3 Surface prior regularisation
 *   #5 Analytical MAP inner loop (diagonal per-band covariance)
 */
#pragma once

#define SM_N_COMPONENTS 3   /* vegetation (0), soil (1), water (2) */

/* Opaque handle; create with surface_model_alloc(), free with surface_model_free(). */
typedef struct SurfaceModelImpl_ SurfaceModelImpl;

/* Allocate and initialise the surface model for 'n_bands' sensor bands.
 * wl_um: array of band-centre wavelengths in µm [n_bands].
 * Returns NULL on allocation failure. */
SurfaceModelImpl *surface_model_alloc(const float *wl_um, int n_bands);

/* Classify a single pixel spectrum to the nearest component.
 * refl: [n_bands] reflectance values (NaN-safe).
 * Uses VNIR bands (0.40–1.00 µm) only for robustness.
 * Returns component index 0–2. */
int surface_model_classify(const SurfaceModelImpl *mdl,
                            const float *refl, int n_bands);

/* MAP regularisation of the full reflectance cube in-place.
 *
 * Layout: refl_cube[b * npix + p] = reflectance of pixel p at band b
 *         sigma2  [b * npix + p] = observation variance (or NULL)
 *
 * When sigma2 == NULL the prior weight 'weight' (0 < weight <= 1) controls
 * the blend: weight=0.1 → 90 % observation, 10 % prior.
 *
 * Algorithm (per pixel, per band, diagonal MAP):
 *   r_map = (r_obs / σ_obs² + r_prior / σ_prior²)
 *           / (1/σ_obs² + 1/σ_prior²)
 *
 * OpenMP-parallelised over pixels. */
void surface_model_regularize(const SurfaceModelImpl *mdl,
                               float *refl_cube,
                               const float *sigma2,
                               int n_bands, int npix,
                               float weight);

/* Per-band model discrepancy added in quadrature to sigma2.
 * wl_um: [n_wl]; sigma_out: [n_wl] output (reflectance units).
 * Baseline 0.5% with bumps at gas-absorption band edges. */
void surface_model_discrepancy(const float *wl_um, int n_wl, float *sigma_out);

void surface_model_free(SurfaceModelImpl *mdl);
