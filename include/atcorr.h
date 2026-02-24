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

/* ─── Version info ────────────────────────────────────────────────────────── */
const char *atcorr_version(void);

#ifdef __cplusplus
}
#endif
