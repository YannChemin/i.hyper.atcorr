/* BRDF surface models — C port of 6SV2.1 *BRDF.f subroutines.
 *
 * Provides a single-point evaluation API for the bidirectional reflectance
 * models available in 6SV2.1.  All models are dispatched through
 * sixs_brdf_eval() using the BrdfType enum.
 */
#pragma once

/* ─── BRDF model identifiers ──────────────────────────────────────────────── */
typedef enum {
    BRDF_LAMBERTIAN    = 0,   /* Lambertian (constant reflectance)       */
    BRDF_RAHMAN        = 1,   /* Rahman-Pinty-Verstraete (RPV)           */
    BRDF_ROUJEAN       = 2,   /* Roujean volumetric kernel               */
    BRDF_HAPKE         = 3,   /* Hapke canopy model                      */
    BRDF_OCEAN         = 4,   /* Ocean (Cox-Munk glint + water body)     */
    BRDF_WALTHALL      = 5,   /* Walthall polynomial                     */
    BRDF_MINNAERT      = 6,   /* Minnaert                                */
    BRDF_VERSFELD      = 7,   /* Verstraete-Pinty (requires mvbp1; stub) */
    BRDF_IAPI          = 8,   /* Iaquinta-Pinty canopy (stub)            */
    BRDF_ROSSLIMAIGNAN = 9,   /* Ross-Li (thick) + Maignan hot-spot      */
} BrdfType;

/* ─── Per-model parameters ─────────────────────────────────────────────────
 * Only the member corresponding to the chosen BrdfType is used. */
typedef union {
    /* BRDF_LAMBERTIAN: rho0 is the constant surface reflectance */
    struct { float rho0; } lambertian;

    /* BRDF_RAHMAN (RPV): rho0 reflectance intensity, af asymmetry, k structure */
    struct { float rho0, af, k; } rahman;

    /* BRDF_ROUJEAN: k0 isotropic, k1 geometric, k2 volumetric kernels */
    struct { float k0, k1, k2; } roujean;

    /* BRDF_HAPKE: om single-scatter albedo, af asymmetry, s0 hotspot amp, h width */
    struct { float om, af, s0, h; } hapke;

    /* BRDF_OCEAN: wspd wind speed (m/s), azw wind azimuth offset (deg),
     *             sal salinity (ppt), pcl chlorophyll (mg/m³), wl wavelength (µm) */
    struct { float wspd, azw, sal, pcl, wl; } ocean;

    /* BRDF_WALTHALL: a, ap, b, c polynomial coefficients */
    struct { float a, ap, b, c; } walthall;

    /* BRDF_MINNAERT: k Minnaert exponent, b albedo normalisation */
    struct { float k, b; } minnaert;

    /* BRDF_ROSSLIMAIGNAN: f_iso, f_vol, f_geo kernel weights */
    struct { float f_iso, f_vol, f_geo; } rosslimaignan;
} BrdfParams;

/* ─── Public API ───────────────────────────────────────────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

/* Evaluate BRDF at a single geometry point.
 *
 * type       : BRDF model
 * params     : model-specific parameters (use appropriate union member)
 * cos_sza    : cos(solar zenith angle)
 * cos_vza    : cos(view zenith angle)
 * raa_deg    : relative azimuth angle (degrees, 0 = forward scatter)
 *
 * Returns ρ_s (dimensionless bidirectional reflectance).
 * Returns 0.0 for BRDF_LAMBERTIAN (Lambertian coupling is handled analytically
 * in atcorr_invert; set params->lambertian.rho0 for the reflectance value). */
float sixs_brdf_eval(BrdfType type, const BrdfParams *params,
                     float cos_sza, float cos_vza, float raa_deg);

/* Compute directional-hemispherical (spherical) albedo by numerical
 * integration of sixs_brdf_eval() over the outgoing hemisphere.
 *
 * n_phi   : number of azimuth quadrature points (suggested: 48)
 * n_theta : number of zenith quadrature points  (suggested: 24)
 *
 * Returns the white-sky albedo ρ̄. */
float sixs_brdf_albe(BrdfType type, const BrdfParams *params,
                     float cos_sza, int n_phi, int n_theta);

#ifdef __cplusplus
}
#endif
