/* i.hyper.atcorr — 6SV2.1-based atmospheric correction library.
 * Public API (importable from Python via ctypes). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Aerosol model identifiers (matching 6SV iaer codes) */
#define AEROSOL_NONE         0
#define AEROSOL_CONTINENTAL  1
#define AEROSOL_MARITIME     2
#define AEROSOL_URBAN        3
#define AEROSOL_DESERT       5

/* Atmosphere model identifiers */
#define ATMO_US62            1   /* US standard 1962 */
#define ATMO_MIDSUM          2   /* Mid-latitude summer */
#define ATMO_MIDWIN          3   /* Mid-latitude winter */
#define ATMO_TROPICAL        4   /* Tropical */
#define ATMO_SUBSUM          5   /* Sub-arctic summer */
#define ATMO_SUBWIN          6   /* Sub-arctic winter */

/* ─── LUT API ─────────────────────────────────────────────────────────────── */

/* 6SV LUT grid specification */
typedef struct {
    /* Wavelength grid: n_wl values in µm */
    const float *wl;        /* [n_wl] */
    int          n_wl;

    /* AOD at 550nm grid */
    const float *aod;       /* [n_aod] */
    int          n_aod;

    /* Column water vapour grid (g/cm²) */
    const float *h2o;       /* [n_h2o] */
    int          n_h2o;

    /* Geometry */
    float sza;              /* solar zenith angle (degrees) */
    float vza;              /* view zenith angle (degrees) */
    float raa;              /* relative azimuth angle (degrees) */

    /* Altitude of surface (km, for plane-level computations) */
    float altitude_km;

    /* Atmosphere and aerosol model */
    int atmo_model;         /* ATMO_* constant */
    int aerosol_model;      /* AEROSOL_* constant */

    /* Surface pressure (hPa, 0 = use standard atmosphere) */
    float surface_pressure;

    /* Ozone column (Dobson units, 0 = standard atmosphere) */
    float ozone_du;
} LutConfig;

/* LUT output arrays — all have size [n_aod × n_h2o × n_wl] (C order: aod, h2o, wl)
 * Accessing: idx = iaod*n_h2o*n_wl + ih2o*n_wl + iwl */
typedef struct {
    float *R_atm;    /* atmospheric path reflectance (unitless) */
    float *T_down;   /* total downward transmittance (direct + diffuse) */
    float *T_up;     /* total upward transmittance (direct + diffuse) */
    float *s_alb;    /* spherical albedo of atmosphere */
} LutArrays;

/* Compute a full LUT.
 * out->R_atm, T_down, T_up, s_alb must be pre-allocated with
 *   n_aod × n_h2o × n_wl floats each.
 * Uses OpenMP for parallelisation over AOD grid points.
 * Returns 0 on success, negative error code on failure. */
int atcorr_compute_lut(const LutConfig *cfg, LutArrays *out);

/* ─── Single-point atmospheric correction ─────────────────────────────────── */

/* Correct a single TOA reflectance pixel.
 * rho_toa: TOA reflectance at given wavelength
 * R_atm, T_down, T_up, s_alb: pre-interpolated LUT values
 * Returns surface (BOA) reflectance. */
static inline float atcorr_invert(float rho_toa, float R_atm,
                                   float T_down, float T_up, float s_alb)
{
    float y = (rho_toa - R_atm) / (T_down * T_up + 1e-10f);
    return y / (1.0f + s_alb * y + 1e-10f);
}

/* ─── Solar irradiance and Earth-Sun distance ─────────────────────────────── */

/* Solar irradiance E0 in W/(m² µm) at wl_um, from 6SV2.1 Thuillier spectrum.
 * Linearly interpolated; clamped at 0.25–4.0 µm. */
float sixs_E0(float wl_um);

/* Squared Earth-Sun distance d² in AU for day-of-year doy (1–365).
 * d = 1 − 0.01670963 × cos(2π(doy−3)/365). */
double sixs_earth_sun_dist2(int doy);

/* ─── LUT spectral slice ──────────────────────────────────────────────────── */

/* Fill per-wavelength correction arrays [n_wl] by bilinear interpolation of
 * the LUT at a fixed (aod_val, h2o_val) point.  Useful for pre-computing
 * scene-average atmospheric parameters before pixel-level inversion.
 *
 * Rs, Tds, Tus, ss must be pre-allocated with cfg->n_wl floats each. */
void atcorr_lut_slice(const LutConfig *cfg, const LutArrays *lut,
                      float aod_val, float h2o_val,
                      float *Rs, float *Tds, float *Tus, float *ss);

/* ─── SRF convolution correction ─────────────────────────────────────────── */

/* Configuration for per-band Gaussian SRF gas-transmittance correction.
 * Used by atcorr_srf_compute() to build a correction table that replaces the
 * coarse 6SV Curtis-Godson gas parameterisation with libRadtran reptran fine
 * (~0.05 nm) gas transmittance convolved with the sensor SRF. */
typedef struct {
    /* Per-band FWHM in µm [n_wl], aligned with LutConfig.wl[].
     * NULL → treat all bands as sub-threshold (correct everything). */
    const float *fwhm_um;

    /* Only correct bands where fwhm_um[i] < threshold_um.
     * 0 or negative → use default of 0.005 µm (5 nm). */
    float threshold_um;
} SrfConfig;

/* Opaque correction table returned by atcorr_srf_compute(). */
typedef struct SrfCorrection_ SrfCorrection;

/* Compute per-band, per-H2O correction factors from libRadtran reptran fine.
 *
 * Runs 4 × n_h2o uvspec subprocesses (fine/coarse × down/up × each H2O)
 * parallelised with OpenMP.  libRadtran must be installed; the uvspec binary
 * is located via PATH, $LIBRADTRAN_DIR/bin, or $GISBASE/bin.  Data files
 * are found via $LIBRADTRAN_DATA or standard paths.
 *
 * Returns NULL if uvspec is not found or no bands fall below threshold_um. */
SrfCorrection *atcorr_srf_compute(const SrfConfig   *srf_cfg,
                                   const LutConfig   *lut_cfg);

/* Apply the correction table to a LUT in place.
 * Must be called after atcorr_compute_lut() and before pixel inversion.
 * Multiplies T_down[ia,ih,iw] and T_up[ia,ih,iw] by the H2O-matched
 * correction factors for each wavelength band. */
void atcorr_srf_apply(const SrfCorrection *srf,
                       const LutConfig     *cfg,
                       LutArrays           *lut);

/* Release memory allocated by atcorr_srf_compute(). */
void atcorr_srf_free(SrfCorrection *srf);

/* ─── Version info ────────────────────────────────────────────────────────── */
const char *atcorr_version(void);

#ifdef __cplusplus
}
#endif
