## DESCRIPTION

*i.hyper.atcorr* is a GRASS GIS add-on for atmospheric correction of
hyperspectral imagery. It operates in two complementary modes that can
be combined in a single invocation:

| Mode | Trigger | Purpose |
|------|---------|---------|
| **LUT generation** | `lut=` | Compute a binary look-up table of atmospheric parameters over an [AOD × H₂O × wavelength] grid |
| **Cube correction** | `input=` / `output=` | Apply the LUT to a Raster3D radiance cube, writing a surface (BOA) reflectance cube |

The LUT is computed using a C port of the 6SV2.1 (Second Simulation of
a Satellite Signal in the Solar Spectrum, version 2.1) radiative
transfer code with OpenMP parallelisation over the AOD dimension.

For each (AOD, H₂O, λ) grid point the module stores four atmospheric
parameters:

- **R_atm** — atmospheric path reflectance
- **T_down** — total downward transmittance (direct + diffuse)
- **T_up** — total upward transmittance (direct + diffuse)
- **s_alb** — spherical albedo of the atmosphere

These four parameters are sufficient to invert a top-of-atmosphere (TOA)
reflectance to surface (BOA) reflectance via the standard algebraic
formula:

```
ρ_toa = (π × L × d²) / (E₀ × cos θₛ)
ρ_boa = (ρ_toa − R_atm) / (T_down × T_up × (1 + s_alb × ρ_boa))
```

where *L* is TOA radiance in W/(m² sr µm), *E₀* is the Thuillier solar
irradiance from 6SV2.1 tables, and *d²* is the squared Earth-Sun
distance for the acquisition day-of-year.

---

## ISOFIT-INSPIRED IMPROVEMENTS

Six improvements inspired by the
[ISOFIT](https://github.com/isofit/isofit) framework are available as
optional flags and parameters. They can be combined freely; each
defaults to disabled for full backward compatibility.

### #1 — Per-pixel atmospheric maps + spatial smoothing

```
aod_map=<raster>   h2o_map=<raster>   smooth=<sigma_px>
```

Supply 2-D raster maps of AOD (at 550 nm) and/or column water vapour
(g/cm²) derived from Dark Target, MAIAC, or any retrieval product.
Per-pixel values replace the scalar `aod_val=` / `h2o_val=` fallbacks;
each pixel is corrected using trilinear LUT interpolation at its own
`(aod, h2o, λ)` point.

`smooth=` applies a **separable Gaussian filter** (σ in pixels) to the
maps before correction, suppressing retrieval noise while preserving
large-scale spatial gradients. Boundary pixels use edge-replication
padding; NaN/null pixels are excluded from the neighbourhood average.

### #2 — In-loop adjacency effect correction

```
adj_psf=<km>   [pixel_size=<m>]
```

Applies the **Vermote et al. (1997) adjacency correction** per band
immediately after algebraic inversion, before spectral regularisation.
The diffuse transmittance fraction carries signal from a spatially-
averaged environmental neighbourhood reflectance:

```
T_diff = clip(T_scat − T_dir, 0, T_scat)
r_env  = box_filter(r_boa, radius = adj_psf / pixel_size)
r_boa += T_diff × s_alb × (r_boa − r_env) / (1 − s_alb × r_env)
```

`T_dir` is the Beer-Lambert two-way direct transmittance computed from
Rayleigh + aerosol optical depths. `pixel_size=` is auto-detected from
the GRASS computational region if not given.

### #3 & #5 — Surface prior MAP regularisation (`-r` flag)

```
-r
```

After all bands have been inverted, blends each pixel's retrieved
spectrum with a **3-component Gaussian mixture surface prior**
(vegetation, soil, water) using diagonal-covariance MAP estimation:

```
r_map[b] = (r_obs[b] / σ_obs²[b]  +  r_prior[b] / σ_prior²[b])
           / (1/σ_obs²[b]  +  1/σ_prior²[b])
```

Each pixel is classified to the nearest component using VNIR bands
(0.40–1.00 µm). Prior means are hardcoded reference spectra
interpolated to the sensor wavelengths; prior variances scale as
`(scale × mean)²` (vegetation 15 %, soil 20 %, water 3 %). The `-r`
flag requires loading the full reflectance cube into memory (~400 MB
for a 426-band Tanager scene).

### #4 — Per-band reflectance uncertainty (`-u` flag)

```
-u   [uncertainty=<output_raster3d>]
```

Computes per-pixel reflectance uncertainty (σ_rfl) for each band from
two propagated sources:

1. **Instrument noise**: NEDL estimated from the standard deviation of
   the darkest 5 % of radiance pixels → propagated as
   `σ_noise = π × NEDL × d² / (E₀ × cos θₛ × T_total)`

2. **AOD uncertainty**: LUT evaluated at `aod ± 0.04`; half the
   reflectance difference gives `σ_aod` per pixel.

Total: `σ_rfl = √(σ_noise² + σ_aod²)`

When `-r` is also enabled, the uncertainty cube feeds the MAP
regularisation as `σ_obs²`.

### #6 — Model discrepancy noise floor

When both `-u` and `-r` are active, a **per-band model discrepancy**
term is added in quadrature to σ_rfl before MAP regularisation.
It reflects systematic RT model errors:

- Baseline: 0.5 % (all bands)
- Gas absorption band edges (720, 760, 940, 1135, 1380, 1850 nm):
  +2 % Gaussian bump (σ = 30 nm)
- SWIR > 1500 nm: +1 % (aerosol model uncertainty)

This prevents over-regularisation in bands where the 6SV gas
parameterisation is least accurate.

---

## C LIBRARY API (libatcorr.so)

`libatcorr.so` exposes the full 6SV2.1 radiative transfer engine as a C
library.  The following functions were added in the Phase 1–5 port from
the original Fortran code and are available to any caller that links against
the library.

### Atmosphere profiles

Six standard atmosphere models are now fully implemented (US Standard 1962,
Mid-latitude Summer, Mid-latitude Winter, Tropical, Sub-arctic Summer,
Sub-arctic Winter).  Select via `atmosphere=` as described above.

### Utility functions (Phase 2)

```c
/* Solar zenith and azimuth from date / location / time */
void sixs_possol(int month, int jday, float tu,
                 float xlon, float xlat,
                 float *asol, float *phi0, int ia);

/* Chandrasekhar analytical Rayleigh reflectance */
float sixs_chand(float xphi, float xmuv, float xmus, float xtau);

/* Vermote environmental adjacency weighting factors */
void sixs_enviro(float difr, float difa, float r, float palt,
                 float xmuv,
                 float *fra, float *fae, float *fr);
```

### Aerosol vertical profile and surface pressure (Phase 3)

```c
/* Fill ctx->aerprof with an exponential aerosol layer profile */
void sixs_aeroprof(SixsCtx *ctx, float aod550, float ome,
                   float haa, float alt);

/* Trim the atmosphere profile to a surface pressure (hPa > 0)
   or altitude (km, passed as negative value) */
void sixs_pressure(SixsCtx *ctx, float sp);

/* As above but also returns the trimmed H2O and O3 columns */
void sixs_pressure_columns(SixsCtx *ctx, float sp,
                            float *uw, float *uo3);
```

### Custom Mie aerosol (Phase 4)

```c
/* Compute aerosol single-scattering properties for a log-normal
   size distribution via Bohren–Huffman Mie scattering.
   Fills ctx->aer (ext/sca/phase at 20 reference wavelengths). */
void sixs_mie_init(SixsCtx *ctx,
                   double r_mode,   /* mode radius µm         */
                   double sigma_g,  /* geometric std dev       */
                   double m_r_550,  /* real refractive index   */
                   double m_i_550); /* imaginary ref. index    */
```

Set `LutConfig.aerosol_type = AEROSOL_CUSTOM` (value 9) and call
`sixs_mie_init()` on the per-thread context before computing the LUT
to use a custom particle size distribution.

Typical continental mineral dust values: `r_mode = 0.07 µm`,
`sigma_g = 2.0`, `m_r = 1.53`, `m_i = 0.008`.

### BRDF surface models (Phase 5) and BRDF-RT coupling

```c
#include "brdf.h"

/* Evaluate BRDF at a single (SZA, VZA, RAA) point */
float sixs_brdf_eval(BrdfType type, const BrdfParams *params,
                     float cos_sza, float cos_vza, float raa_deg);

/* Bihemispherical (white-sky) albedo by midpoint quadrature over the
 * outgoing hemisphere.  n_phi=48, n_theta=24 gives < 0.1 % error. */
float sixs_brdf_albe(BrdfType type, const BrdfParams *params,
                     float cos_sza, int n_phi, int n_theta);
```

Available models (`BrdfType` enum):

| Value | Name | Description |
|-------|------|-------------|
| 0 | `BRDF_LAMBERTIAN` | Isotropic Lambertian (single `rho0` parameter) |
| 1 | `BRDF_RAHMAN` | Rahman–Pinty–Verstraete (RPV) — ρ₀, k, Θ |
| 2 | `BRDF_ROUJEAN` | Roujean volumetric kernel — k0, k1, k2 |
| 3 | `BRDF_HAPKE` | Hapke canopy model — ω, b, c, h |
| 4 | `BRDF_OCEAN` | Cox–Munk glint + whitecaps + water body — wind speed, pigment, salinity, foam fraction, water reflectance |
| 5 | `BRDF_WALTHALL` | Walthall polynomial — a0, a1, a2, a3 |
| 6 | `BRDF_MINNAERT` | Minnaert — k (limb-darkening exponent) |
| 7 | `BRDF_VERSFELD` | Versfeld (stub — returns 0) |
| 8 | `BRDF_IAPI` | IAPI canopy (stub — returns 0) |
| 9 | `BRDF_ROSSLIMAIGNAN` | Ross–Li + Maignan hot-spot — fiso, fvol, fgeo, h_ratio |

**BRDF-RT coupling** is enabled via the `brdf=` and `brdf_params=`
GRASS options (see Parameters section).  DISCOM always runs with a black
lower boundary (correct and efficient); the BRDF model is applied in the
inversion formula only.  For Lambertian surfaces the standard
`atcorr_invert()` path is unchanged.  For non-Lambertian surfaces
`atcorr_invert_brdf()` is used:

```
y        = (ρ_toa − R_atm) / (T_down × T_up)
ρ_brdf   = y × (1 − s_alb × ρ̄)
```

where ρ̄ is the bihemispherical (white-sky) albedo computed once per scene
via 48 × 24 midpoint quadrature.  The output is the bidirectional
reflectance factor (BRF) at the acquisition geometry.

---

## BRDF CORRECTION OPTIONS

```
brdf=lambertian|rahman|roujean|hapke|ocean|walthall|minnaert|rosslimaignan
brdf_params=<comma-separated floats>
```

Select a non-Lambertian surface BRDF model for the inversion formula.
`brdf_params=` accepts up to 5 comma-separated floats whose meaning
depends on the chosen model:

| Model | Parameters |
|-------|-----------|
| `lambertian` | `rho0` (constant reflectance; default 1.0) |
| `rahman` | `rho0, af, k` |
| `roujean` | `k0, k1, k2` |
| `hapke` | `om, af, s0, h` |
| `ocean` | `wspd_ms, azw_deg, sal_ppt, pcl_mgl, wl_um` |
| `walthall` | `a, ap, b, c` |
| `minnaert` | `k, b` |
| `rosslimaignan` | `f_iso, f_vol, f_geo` |

When `brdf=lambertian` (default), the standard Lambertian `atcorr_invert()`
formula is used and the output is BOA reflectance.  For any other model,
`atcorr_invert_brdf()` is used and the output is the bidirectional
reflectance factor (BRF) at the acquisition geometry.

---

## IMAGE-BASED RETRIEVAL OPTIONS

i.hyper.atcorr can estimate the atmospheric state parameters directly from
the input hyperspectral cube, making the module fully standalone for scenes
with no external ancillary data.  All four retrievals are optional and
independent; they run before LUT computation so that retrieved O₃ and
surface pressure update the LUT, while retrieved AOD and H₂O are used as
per-pixel correction maps.

### `-z` — O₃ column from Chappuis band depth

Estimates the scene-mean ozone column from the Chappuis absorption feature
at ~600 nm using a continuum-interpolation approach:

```
L_cont = linear_interp(L_540, L_680) at 600 nm
D      = max(0, 1 − L_600 / L_cont)
O₃     = D / (σ₆₀₀ × m)     σ₆₀₀ ≈ 1.0×10⁻⁴ DU⁻¹
```

where *m* = 1/cos(θₛ) + 1/cos(θᵥ) is the two-way airmass factor.
The result (scene-mean DU) replaces the `ozone=` value for LUT computation.
Requires input= and band wavelength metadata in the cube.

### `-w` — Column water vapour from 940 nm band depth

Estimates per-pixel water vapour column [g/cm²] from the 940 nm absorption
feature using the **Kaufman & Gao (1992)** continuum interpolation:

```
L_cont = linear_interp(L_865, L_1040) at 940 nm
D      = max(0, 1 − L_940 / L_cont)
WVC    = D / (K₉₄₀ × m)     K₉₄₀ = 0.036 cm²/g
```

The per-pixel WVC map replaces `h2o_val=` as the correction input.
The scene mean updates `h2o_val=` as the scalar fallback.
Requires input= and band wavelength metadata.

### `-a` — AOD from Dark Dense Vegetation (DDV)

Estimates per-pixel AOD at 550 nm from dark vegetated pixels using the
**MODIS dark-target approach** (Kaufman et al. 1997):

**DDV mask:** 0.01 < ρ_toa(2130) < 0.25 AND NDVI(860, 660) > 0.1

**Surface prediction:** ρ_surf_470 = 0.25 ρ_2130; ρ_surf_660 = 0.50 ρ_2130

**Single-scattering inversion** (nadir view, Henyey-Greenstein, g=0.65, ω₀=0.89):

```
ρ_path = max(0, ρ_toa − ρ_surf)
τ      = ρ_path × 4μₛ / (ω₀ × P_HG)
α      = −ln(τ_470/τ_660) / ln(470/660)    (Ångström exponent)
τ_550  = τ_470 × (550/470)^(−α)
```

Non-DDV pixels are filled with the scene-mean AOD.  Requires bands at
approximately 470, 660, 860, and 2130 nm.

### `dem=` — Surface pressure from terrain elevation

Computes the scene-mean terrain elevation from a 2-D DEM raster [m] and
converts it to surface pressure using the ISA barometric formula:

```
P = 1013.25 × (1 − 2.2558×10⁻⁵ × h)^5.2559   [hPa]
```

The result replaces the standard-atmosphere sea-level pressure in LUT
computation, improving accuracy for elevated terrain.

### C Library API (retrieve.h)

The retrieval functions are also available from `libatcorr.so`:

```c
#include "retrieve.h"

/* H2O column [g/cm²] per pixel from 940 nm band depth */
void retrieve_h2o_940(const float *L_865, const float *L_940,
                       const float *L_1040, int npix,
                       float sza_deg, float vza_deg,
                       float *out_wvc);

/* Per-pixel AOD at 550 nm from DDV; returns scene mean */
float retrieve_aod_ddv(const float *L_470, const float *L_660,
                        const float *L_860, const float *L_2130,
                        int npix, int doy, float sza_deg,
                        float *out_aod);

/* Scene-mean O₃ [DU] from Chappuis band depth */
float retrieve_o3_chappuis(const float *L_540, const float *L_600,
                            const float *L_680, int npix,
                            float sza_deg, float vza_deg);

/* ISA pressure [hPa] from mean terrain elevation [m] */
float retrieve_pressure_isa(float elev_m);
```

All functions are pure computation with no GRASS dependencies and can be
called directly from Python via ctypes.

---

## NOTES

### LUT file format

The binary file is host-endian (little-endian on x86) and has the
following layout:

```
magic     uint32   0x4C555400 ("LUT\0")
version   uint32   1
n_aod     int32
n_h2o     int32
n_wl      int32
aod[n_aod]          float32
h2o[n_h2o]          float32
wl [n_wl]           float32   (micrometres)
R_atm [n_aod*n_h2o*n_wl]   float32
T_down[n_aod*n_h2o*n_wl]   float32
T_up  [n_aod*n_h2o*n_wl]   float32
s_alb [n_aod*n_h2o*n_wl]   float32
```

Array indexing is C order: the wavelength index varies fastest.
Typical size: ~4 MB for 6 AOD × 5 H₂O × 211 wavelengths.

### Radiative transfer

The RT solver is a direct C port of the 6SV2.1 Fortran code (Vermote
et al. 1997). Scattering is computed via the Successive Orders of
Scattering (SOS) method with Gauss–Legendre quadrature (25 points per
hemisphere, 30 atmospheric layers). Gas absorption (H₂O, O₃, CO₂, O₂,
N₂O, CH₄) is computed separately and folded into T_down and T_up so
that the scattering LUT only needs to be computed once per AOD value.

### OpenMP parallelism

Each AOD grid point is independent and is computed in a separate OpenMP
thread. On a 4-core machine a 6 AOD × 5 H₂O × 211 band LUT takes
approximately one second. The Gaussian spatial filter, adjacency
correction, uncertainty computation, and MAP regularisation all use
additional OpenMP parallel regions.

### Cube correction memory

When `-r` is active, the full reflectance cube is held in memory for
MAP regularisation. Memory usage is approximately
`n_bands × n_pixels × 4` bytes. For a 426-band Tanager scene with a
typical 640 × 480 tile this is about 400 MB. When `-r` is not set,
bands are written to the output Raster3D immediately after inversion.

### Band wavelength metadata

When correcting a Raster3D cube the module auto-reads band wavelengths
from `r3.info -h` metadata in the format `Band N: WL nm`. Make sure
this metadata is present in the input cube.

---

## EXAMPLES

Each example below first generates a LUT and then applies it to correct
a Raster3D radiance cube. Scene geometry, atmosphere, and aerosol type
are chosen to reflect realistic acquisition conditions for the land cover
and climate zone described.

**Image-based retrieval quick reference** — which flags to activate per scene:

| Example | Flags | Rationale |
|---------|-------|-----------|
| 1. Saharan dust | `-z dem=` | No DDV (barren desert); dry uniform H₂O; O₃ and elevation matter |
| 2. Amazon | `-z -w -a` | Dense forest = ideal DDV; ~4 g/cm² WVC gradient; low tropical O₃ |
| 3. Paris winter | `-z -a` | Variable O₃ (polar vortex); farmland DDV; dry winter air |
| 4. Mediterranean | `-w -a` | Land–sea H₂O gradient; coastal DDV; stable O₃ |
| 5. Sweden winter | `-z` | Polar O₃ enhancement at SZA=78°; snow = no DDV; very dry |
| 6. Yukon summer | `-z -w -a dem=` | All four: boreal DDV + tundra WVC gradient + variable O₃ + elevation |

---

### 1. Saharan dust storm (desert, tropical atmosphere)

**Scene**: Algeria, March (DOY 90). High-sun geometry, extremely dry air,
heavy mineral dust load (AOD can exceed 2 during events).

```sh
# LUT: wide AOD range to cover moderate and heavy dust episodes
i.hyper.atcorr \
    lut=sahara_dust.lut \
    sza=40 vza=5 raa=150 \
    atmosphere=tropical aerosol=desert ozone=280 \
    aod=0.05,0.2,0.5,1.0,2.0,3.0 \
    h2o=0.2,0.5,1.0,1.5 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005

# Correction with scalar state from an external MODIS dust product
i.hyper.atcorr \
    input=algeria_radiance output=algeria_refl \
    lut=sahara_dust.lut \
    sza=40 vza=5 raa=150 doy=90 \
    aod_val=0.85 h2o_val=0.4 \
    atmosphere=tropical aerosol=desert

# Image-based O₃ retrieval (-z) + ISA pressure from DEM.
# -a omitted: barren desert has no DDV-eligible vegetation pixels.
# -w omitted: Saharan H₂O < 1.5 g/cm² is spatially uniform; scalar sufficient.
i.hyper.atcorr -z \
    input=algeria_radiance output=algeria_refl \
    lut=sahara_dust_auto.lut \
    sza=40 vza=5 raa=150 doy=90 \
    atmosphere=tropical aerosol=desert \
    dem=srtm_dem \
    aod=0.05,0.2,0.5,1.0,2.0,3.0 h2o=0.2,0.5,1.0,1.5 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005
```

**Why `atmosphere=tropical`**: The tropical profile has the highest
near-surface temperature and pressure, better matching hot Saharan
conditions than the US62 standard.  Desert aerosol uses the Shettle/Fenn
mineral dust optical model (strongly wavelength-dependent: high AOD in
blue, flatter in NIR).  The narrow H₂O grid (0.2–1.5 g/cm²) reflects
the extremely low precipitable water of the Sahara.

**Retrieval flags**: **`-z`** retrieves scene-mean O₃ from the Chappuis
absorption band centred at 600 nm.  Tropical stratospheric O₃ deviates
±30–50 DU from the 280 DU prior, introducing a systematic tilt in the
540–700 nm reflectance if uncorrected.  **`dem=`** replaces the default
1013.25 hPa with the ISA formula at the mean terrain elevation; elevated
desert plateaux (e.g. Hoggar massif, ~2 000 m, P ≈ 795 hPa) have ~20 %
lower Rayleigh optical depth than sea level.  **`-a`** and **`-w`** are
omitted: barren desert has no DDV vegetation, and Saharan WVC is spatially
uniform enough for the scalar prior.

---

### 2. Tropical rainforest (Amazon, high humidity)

**Scene**: Pará state, Brazil, September (DOY 270). Near-overhead sun,
very high water vapour, background biogenic aerosol.

```sh
i.hyper.atcorr \
    lut=amazon.lut \
    sza=28 vza=0 raa=0 \
    atmosphere=tropical aerosol=continental ozone=260 \
    aod=0.03,0.08,0.15,0.30,0.50 \
    h2o=3.5,4.5,5.5,6.5,7.5 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005

# Correction with scalar state (typical wet-season baseline)
i.hyper.atcorr \
    input=amazon_radiance output=amazon_refl \
    lut=amazon.lut \
    sza=28 vza=0 raa=0 doy=270 \
    aod_val=0.10 h2o_val=5.5 \
    atmosphere=tropical aerosol=continental

# Image-based retrieval: per-pixel H₂O (-w) + per-pixel AOD (-a) + O₃ (-z).
# Dense tropical forest is an ideal DDV target (NDVI > 0.8, ρ_2130 ≈ 0.05–0.12).
# H₂O gradients across the basin can span 3–7 g/cm²; per-pixel retrieval
# from the 940 nm band removes this spatial bias.
i.hyper.atcorr -z -w -a \
    input=amazon_radiance output=amazon_refl \
    lut=amazon_auto.lut \
    sza=28 vza=0 raa=0 doy=270 \
    atmosphere=tropical aerosol=continental \
    aod=0.03,0.08,0.15,0.30,0.50 h2o=3.5,4.5,5.5,6.5,7.5 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005
```

**Why high H₂O grid**: Precipitable water in tropical Amazonia routinely
exceeds 5 g/cm².  Extending the H₂O LUT axis above the default 5 g/cm²
prevents extrapolation error in the 940 nm and 1135 nm water vapour bands,
where transmittance is highly nonlinear.  Continental aerosol represents
the Aitken-mode biogenic VOC particles typical of the clean wet season.

**Retrieval flags**: **`-w`** retrieves per-pixel WVC from the 940 nm band
depth; the continental Amazon spans a ~4 g/cm² WVC gradient at any given
overpass, which translates to a 5–12 % NIR reflectance error if a single
scalar is used.  **`-a`** exploits dense forest as a DDV target
(ρ_2130 ≈ 0.05–0.12, NDVI > 0.8); per-pixel AOD from the 470/660 nm pair
captures smoke plumes from fire fronts at the forest edge.  **`-z`**
retrieves the tropical O₃ column, which is ~10–15 DU lower than the
mid-latitude standard (260 DU assumed vs 300 DU default), preventing a
systematic blue/green reflectance bias.

---

### 3. Urban temperate mid-winter (continental pollution)

**Scene**: Paris region, France, January (DOY 15). Low sun, cold and
dry air, traffic and heating fuel combustion aerosol, possible anticyclone
pollution build-up.

```sh
i.hyper.atcorr \
    lut=paris_winter.lut \
    sza=72 vza=5 raa=120 \
    atmosphere=midwin aerosol=urban ozone=330 \
    aod=0.05,0.15,0.30,0.60,1.00 \
    h2o=0.2,0.4,0.7,1.0 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005

# Correction with scalar state + uncertainty output
i.hyper.atcorr -u \
    input=paris_radiance output=paris_refl \
    lut=paris_winter.lut \
    sza=72 vza=5 raa=120 doy=15 \
    aod_val=0.35 h2o_val=0.5 \
    atmosphere=midwin aerosol=urban \
    uncertainty=paris_refl_unc

# Image-based O₃ retrieval (-z) + DDV AOD from suburban farmland (-a).
# -w omitted: Paris in January has H₂O ≈ 0.2–0.7 g/cm²; spatial
# variability is small and the scalar prior is adequate.
i.hyper.atcorr -z -a -u \
    input=paris_radiance output=paris_refl \
    lut=paris_winter_auto.lut \
    sza=72 vza=5 raa=120 doy=15 \
    atmosphere=midwin aerosol=urban \
    uncertainty=paris_refl_unc \
    aod=0.05,0.15,0.30,0.60,1.00 h2o=0.2,0.4,0.7,1.0 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005
```

**Why `atmosphere=midwin` + `aerosol=urban`**: Mid-latitude winter
profiles have lower water vapour scale heights and a strong temperature
inversion that traps pollution.  Urban aerosol contains soot and sulfate
(higher single-scattering absorption, lower SSA than continental), making
the path radiance correction in the visible bands significantly larger
than for clean rural air.  The `-u` flag propagates AOD uncertainty
(±0.04) into the reflectance product, important here because high SZA
inflates the atmospheric path radiance sensitivity.

**Retrieval flags**: **`-z`** retrieves scene-mean O₃ from the Chappuis
band; winter mid-latitude O₃ columns over France range 300–380 DU
depending on quasi-biennial oscillation phase and polar vortex proximity —
a ±40 DU offset biases 540–700 nm reflectance by ~0.5 % at SZA=72°.
**`-a`** retrieves per-pixel AOD using DDV pixels in the agricultural
Beauce plain south of Paris (winter wheat fields, NDVI ~0.3–0.5,
ρ_2130 ≈ 0.06–0.12); the urban AOD gradient from city centre to rural
fringe spans 0.1–0.5, so per-pixel retrieval significantly improves the
correction over suburban areas.  **`-w`** is omitted: mid-winter WVC over
northern France is low (0.2–0.7 g/cm²) and spatially uniform.

---

### 4. Coastal temperate mid-summer (maritime aerosol, adjacency correction)

**Scene**: Gulf of Lion, Mediterranean, July (DOY 200). Low sun elevation
at overpass time is unusual but covered; clean marine air; land-sea boundary
requires adjacency correction.

```sh
i.hyper.atcorr \
    lut=mediterranean.lut \
    sza=22 vza=0 raa=0 \
    atmosphere=midsum aerosol=maritime ozone=315 \
    aod=0.02,0.06,0.12,0.20,0.30 \
    h2o=1.5,2.5,3.5,4.5 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005

# Per-pixel atmospheric maps from external products + adjacency correction
i.hyper.atcorr -u -r \
    input=med_radiance output=med_refl \
    lut=mediterranean.lut \
    sza=22 vza=0 raa=0 doy=200 \
    atmosphere=midsum aerosol=maritime \
    aod_map=maiac_aod h2o_map=mod05_wvc \
    smooth=2 \
    adj_psf=0.5 \
    uncertainty=med_refl_unc

# Image-based alternative: per-pixel H₂O (-w) + DDV AOD (-a) from coastal
# vegetation replace the external MAIAC and MOD05 maps.
# -z omitted: Mediterranean O₃ (310–320 DU) is stable; scalar reliable.
i.hyper.atcorr -w -a -u -r \
    input=med_radiance output=med_refl \
    lut=med_auto.lut \
    sza=22 vza=0 raa=0 doy=200 \
    atmosphere=midsum aerosol=maritime \
    smooth=2 adj_psf=0.5 \
    uncertainty=med_refl_unc \
    aod=0.02,0.06,0.12,0.20,0.30 h2o=1.5,2.5,3.5,4.5 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005
```

**Why adjacency correction matters here**: At a land-sea boundary the
bright land signal bleeds into adjacent water pixels through atmospheric
forward scattering; the diffuse transmittance fraction (~10–20 % for
maritime aerosol at 550 nm) carries the environmental mean reflectance.
`adj_psf=0.5 km` accounts for the typical 500 m scattering radius for
clean marine conditions.  Maritime aerosol has low absorption (SSA ≈ 0.98)
and coarse sea-salt particles that produce strong forward scattering —
the standard deviation of the path radiance is lower than for urban air
but the adjacency radius is larger.

**Retrieval flags**: **`-w`** retrieves per-pixel WVC from 940 nm; the
Mediterranean in July has a strong land–sea H₂O gradient (1.5–4.0 g/cm²
coast to open sea), which satellite products resolve at only 5–10 km —
worse than Tanager's sub-km pixels.  **`-a`** exploits coastal garrigue
and irrigated farmland as DDV targets; the marine AOD horizontal gradient
(0.05–0.20 AOD from open sea to haze near the Rhône delta) is well
captured by per-pixel retrieval.  **`-z`** is omitted: western
Mediterranean mid-summer O₃ (~315 DU) varies by < 10 DU between years,
within the noise of the Chappuis retrieval at moderate SZA.

---

### 5. Sub-arctic mid-winter with full ISOFIT pipeline

**Scene**: Boreal forest, Northern Sweden, January (DOY 15). Very low
sun, snow-covered ground, clean sub-arctic air.

```sh
i.hyper.atcorr \
    lut=sweden_winter.lut \
    sza=78 vza=8 raa=45 \
    atmosphere=subwin aerosol=continental ozone=340 \
    aod=0.01,0.05,0.10,0.20 \
    h2o=0.1,0.3,0.5,0.8 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005

# Full ISOFIT pipeline with external AOD/H₂O maps
i.hyper.atcorr -u -r \
    input=sweden_radiance output=sweden_refl \
    lut=sweden_winter.lut \
    sza=78 vza=8 raa=45 doy=15 \
    atmosphere=subwin aerosol=continental \
    aod_map=maiac_aod h2o_map=mod05_wvc \
    smooth=3 adj_psf=1.0 \
    uncertainty=sweden_refl_unc

# Add image-based O₃ retrieval (-z): sub-arctic winter O₃ varies
# significantly with polar vortex proximity; -a and -w not used
# (snow-covered ground has no DDV pixels; H₂O < 0.5 g/cm² is well-known).
i.hyper.atcorr -z -u -r \
    input=sweden_radiance output=sweden_refl \
    lut=sweden_winter_auto.lut \
    sza=78 vza=8 raa=45 doy=15 \
    atmosphere=subwin aerosol=continental \
    aod_map=maiac_aod h2o_map=mod05_wvc \
    smooth=3 adj_psf=1.0 \
    uncertainty=sweden_refl_unc \
    aod=0.01,0.05,0.10,0.20 h2o=0.1,0.3,0.5,0.8 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005
```

**Why `atmosphere=subwin`**: Sub-arctic winter profiles have the coldest
temperature structure and the lowest H₂O scale height of the six models,
with significant stratospheric ozone enhancement.  At SZA=78° the path
length through the atmosphere is ~5× that at nadir, so precise
atmosphere profile selection matters more than at low sun angles.  The
`-r` MAP regularisation is especially useful here: bright snow reflectance
is nearly spectrally flat (prior: soil component), which constrains
physically implausible spectral spikes caused by residual gas-correction
errors at very long path lengths.

**Retrieval flags**: **`-z`** is particularly valuable at high latitudes in
winter: the polar vortex can push stratospheric O₃ columns to 350–400 DU
(vs the 340 DU prior), and the long path length at SZA=78° amplifies O₃
absorption by ~×5 relative to nadir — a 40 DU offset then causes a ~1 %
error in the green–red bands.  Retrieving O₃ from the Chappuis band
eliminates this latitude-dependent offset without requiring a real-time O₃
product.  **`-a`** is omitted: snow in January has ρ_2130 > 0.25, excluding
virtually all land pixels from the DDV mask.  **`-w`** is omitted: sub-arctic
winter WVC (< 0.5 g/cm²) is well-captured by the narrow H₂O LUT grid; at
SZA=78° even small WVC errors amplify path corrections, so the external
MOD05 map is preferred over the noisier 940 nm retrieval in very cold, dry
conditions.

---

### 6. Sub-arctic mid-summer — boreal forest

**Scene**: Yukon Territory, Canada, July (DOY 190). Low-moderate sun,
moderate water vapour from thawed tundra, clean background aerosol.

```sh
i.hyper.atcorr \
    lut=yukon_summer.lut \
    sza=45 vza=2 raa=60 \
    atmosphere=subsum aerosol=continental ozone=320 \
    aod=0.02,0.07,0.15,0.30 \
    h2o=0.8,1.5,2.5,3.5 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005

# Correction with scalar state (scalar baseline)
i.hyper.atcorr \
    input=yukon_radiance output=yukon_refl \
    lut=yukon_summer.lut \
    sza=45 vza=2 raa=60 doy=190 \
    aod_val=0.08 h2o_val=2.0 \
    atmosphere=subsum aerosol=continental

# Fully standalone — all four atmospheric state variables retrieved from image.
# -z: sub-arctic O₃ varies 300–340 DU with planetary wave activity.
# -w: tundra WVC spans 1–3.5 g/cm²; per-pixel 940 nm retrieval captures gradient.
# -a: dense boreal spruce forest is an ideal DDV target (NDVI ~0.6–0.8).
# dem=: Yukon terrain spans 500–2000+ m; ISA pressure correction is significant.
i.hyper.atcorr -z -w -a \
    input=yukon_radiance output=yukon_refl \
    lut=yukon_summer_auto.lut \
    sza=45 vza=2 raa=60 doy=190 \
    atmosphere=subsum aerosol=continental \
    dem=dem_yukon \
    aod=0.02,0.07,0.15,0.30 h2o=0.8,1.5,2.5,3.5 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005
```

**Why `atmosphere=subsum`**: Sub-arctic summer combines moderate surface
temperatures with relatively dry air (H₂O < 3.5 g/cm²), a profile
distinct from both the tropical and mid-latitude summer models.  The
boreal growing season produces biogenic terpene aerosol well captured by
the continental aerosol model.

**Retrieval flags**: **`-z`** retrieves scene-mean O₃; sub-arctic summer
O₃ varies 300–340 DU with planetary wave activity.  **`-w`** retrieves
per-pixel WVC from the 940 nm band; heterogeneous boreal–tundra landscapes
have WVC gradients of 1–2 g/cm² over 10–20 km (wet valley bogs vs dry
ridgelines).  **`-a`** uses dense spruce forest as a DDV target
(ρ_2130 ≈ 0.05–0.12, NDVI ~0.6–0.8) to retrieve per-pixel AOD; wildfire
smoke episodically raises AOD at 550 nm from background ~0.05 to > 0.5.
**`dem=`** applies the ISA barometric formula to the mean terrain elevation;
a 1 000 m elevation change reduces surface pressure by ~11 %, which if
uncorrected biases Rayleigh scattering by ~3 % in the blue bands.

---

### 7. Custom Mie aerosol + BRDF via the C API (Python)

The GRASS module exposes the five standard aerosol models via the
`aerosol=` flag. For research applications requiring a specific particle
size distribution (e.g. fresh wildfire smoke, mineral dust with a
measured size distribution), use `sixs_mie_init()` through the Python
bindings:

```python
import ctypes, numpy as np
from python.atcorr import LutConfig, LutArrays, compute_lut
from include.brdf import BrdfType, BrdfParams

# -------------------------------------------------------------------
# Example A: Fresh wildfire smoke over boreal forest
# Carbonaceous particles (AERONET SMART-COMMIT median values for boreal
# smoke): small mode, strongly absorbing.
# Combined with Ross-Li-Maignan BRDF for the forest canopy below.
# -------------------------------------------------------------------
cfg = LutConfig()
cfg.aerosol_type   = 9            # AEROSOL_CUSTOM — use Mie init
cfg.mie_r_mode     = 0.08         # mode radius µm (fine carbonaceous)
cfg.mie_sigma_g    = 1.70         # geometric std dev (narrow smoke mode)
cfg.mie_m_real     = 1.52         # real refractive index at 550 nm
cfg.mie_m_imag     = 0.045        # imaginary part: strong absorption (SSA≈0.87)
cfg.atmosphere_type = 5           # subsum — boreal summer
cfg.sza            = 45.0
cfg.vza            = 2.0
cfg.raa            = 60.0
cfg.brdf_type      = BrdfType.BRDF_ROSSLIMAIGNAN
# Ross-Li-Maignan for dense boreal forest (summertime LAI≈4)
cfg.brdf_params    = (0.08, 0.04, 0.01, 2.0, 0, 0)  # fiso,fvol,fgeo,h_ratio
lut = compute_lut(cfg)

# Why this combination?
# 1. Standard aerosol models represent background climatology.
#    Fresh smoke events have much smaller r_mode (~0.06-0.10 µm vs 0.15 µm
#    for continental), higher m_i (0.03-0.06 vs 0.001), and Angstrom
#    exponents of 1.8-2.2 (vs 1.2 for continental). Using the wrong
#    aerosol optical model introduces a systematic positive reflectance
#    bias of 3-8% in the blue bands and a negative bias in the NIR.
# 2. The Ross-Li-Maignan BRDF captures the hotspot reflectance surge
#    of a closed forest canopy (observed ratio of forward/backward
#    scattering ~3:1 in NIR). Lambertian assumption underestimates
#    reflectance at hotspot geometry by 15-25%.
# 3. Combining both corrections simultaneously avoids double-counting:
#    the aerosol correction uses the correct Mie phase function, and
#    the surface correction uses the correct anisotropy factor.

# -------------------------------------------------------------------
# Example B: Saharan dust episode over dune fields
# Coarse calcite/quartz particles measured during a Bodélé event.
# Combined with Hapke BRDF for bright desert sand.
# -------------------------------------------------------------------
cfg2 = LutConfig()
cfg2.aerosol_type  = 9
cfg2.mie_r_mode    = 1.5          # coarse mode radius µm (supermicron dust)
cfg2.mie_sigma_g   = 2.3          # broad distribution (bimodal dust)
cfg2.mie_m_real    = 1.55         # calcite/quartz real part
cfg2.mie_m_imag    = 0.003        # very low imaginary part (nearly conservative)
cfg2.atmosphere_type = 4          # tropical
cfg2.sza           = 40.0
cfg2.vza           = 5.0
cfg2.raa           = 150.0
cfg2.brdf_type     = BrdfType.BRDF_HAPKE
# Hapke for bright quartz sand (omega=0.88, b=0.25, c=0.80, h=0.48)
cfg2.brdf_params   = (0.88, 0.25, 0.80, 0.48, 0, 0)
lut2 = compute_lut(cfg2)

# Why this combination?
# 1. Heavy dust events shift the size distribution toward coarser particles
#    (r_eff > 2 µm) compared to the standard Shettle/Fenn desert model
#    (r_eff ≈ 0.5 µm, tuned for background mineral aerosol). Coarser
#    particles have: lower Angstrom exponent (α≈0.1-0.4 vs 0.4-0.6),
#    stronger forward phase function peak, and larger single-scattering
#    albedo in the blue (less absorption relative to scattering).
#    Mie modelling with the measured size distribution reduces the NIR
#    reflectance bias by 5-12% for thick dust layers (AOD > 0.8).
# 2. Desert sand is one of the most strongly non-Lambertian surfaces.
#    Hapke's photometric model captures the opposition surge (sharp
#    backscatter peak at phase angle ≈ 0°) and the broad forward-
#    scattering lobe. Lambertian assumption underestimates solar-noon
#    reflectance by up to 20% for bright sand (rho_Lamb ≈ 0.45 vs
#    rho_Hapke ≈ 0.54 at hotspot).
```

---

### 8. Quick test at coarse spectral resolution

```sh
i.hyper.atcorr --verbose \
    lut=/tmp/test.lut \
    sza=30 vza=0 raa=0 \
    aod=0.0,0.2,0.4 h2o=2.0,4.0 \
    wl_min=0.4 wl_max=2.5 wl_step=0.05
```

---

### 9. Fully standalone — all atmospheric state from the image

This example demonstrates the autonomous mode where all four atmospheric
state variables are retrieved from the hyperspectral image itself,
eliminating the need for external ancillary products (MODIS O₃, MOD05 WVC,
MAIAC AOD, or DEM pressure).  A single command generates the LUT and
applies the correction.

```sh
# All four image-based retrievals in one pass:
#   -z  → scene-mean O₃ from Chappuis band (540/600/680 nm)
#   -w  → per-pixel WVC from 940 nm band depth (865/940/1040 nm)
#   -a  → per-pixel AOD from MODIS DDV algorithm (470/660/860/2130 nm)
#   dem= → ISA surface pressure from mean DEM elevation
i.hyper.atcorr -z -w -a \
    input=scene_radiance output=scene_refl \
    lut=scene_auto.lut \
    sza=40 vza=3 raa=120 doy=180 \
    atmosphere=midsum aerosol=continental \
    dem=terrain_dem \
    aod=0.0,0.05,0.1,0.2,0.4,0.8 \
    h2o=0.5,1.5,3.0,5.0 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005
```

**What each flag retrieves**:

- **`-z`** — Scene-mean O₃ (Dobson units) from the Chappuis absorption band
  centred at 600 nm.  Continuum: linear interpolation of L_540 and L_680;
  band depth D = 1 − L_600/L_cont; O₃ = D / (σ_600 × m) where
  σ_600 ≈ 10⁻⁴ DU⁻¹ and m = 1/cos(SZA) + 1/cos(VZA).  The retrieved value
  replaces `ozone=` before the LUT is computed; falls back to 300 DU if
  fewer than one valid pixel is found.

- **`-w`** — Per-pixel column water vapour (g/cm²) from the 940 nm band
  depth.  Continuum: linear interpolation of L_865 and L_1040;
  D = 1 − L_940/L_cont; WVC = D / (K_940 × m) where K_940 = 0.036 cm²/g.
  The per-pixel array is used during correction in place of the scalar
  `h2o_val=`; pixels without valid 940 nm signal fall back to 2 g/cm².

- **`-a`** — Per-pixel AOD at 550 nm from the MODIS DDV algorithm.  DDV
  mask: 0.01 < ρ_2130 < 0.25 AND NDVI(860/660) > 0.1.  Surface reflectance
  predicted as ρ_surf_470 = 0.25 ρ_2130 and ρ_surf_660 = 0.50 ρ_2130.
  Ångström exponent from the 470/660 nm pair scales τ to 550 nm.  Non-DDV
  pixels receive the scene-mean AOD.  Requires bands near 470, 660, 860, and
  2130 nm.

- **`dem=`** — Mean terrain elevation → ISA surface pressure
  P = 1013.25 × (1 − 2.2558 × 10⁻⁵ × h)^5.2559 hPa.  Updates
  `surface_pressure` before LUT computation.  Eliminates the systematic
  Rayleigh scattering bias at elevations above ~500 m.

**Prerequisites**: the image must span at least 376–2499 nm with adequate
SNR at 540/600/680 nm (O₃), 865/940/1040 nm (H₂O), and 470/660/860/2130 nm
(AOD).  Tanager, DESIS, HySpex, PRISMA, and EnMAP all satisfy these
requirements.

---

## SEE ALSO

*[i.hyper.smac](i.hyper.smac.md), [i.atcorr](i.atcorr.md)*

## REFERENCES

- Vermote, E.F., Tanré, D., Deuzé, J.L., Herman, M. and Morcrette, J.J.
  (1997): Second simulation of the satellite signal in the solar
  spectrum, 6S: An overview. *IEEE Transactions on Geoscience and
  Remote Sensing*, 35(3), 675–686.
- Kotchenova, S.Y., Vermote, E.F., Matarrese, R. and Klemm, F.J. (2006):
  Validation of a vector version of the 6S radiative transfer code for
  atmospheric correction of satellite data. *Applied Optics*, 45(26),
  6762–6778.
- Thompson, D.R. et al. (2018): Optimal estimation for imaging
  spectrometer atmospheric correction. *Remote Sensing of Environment*,
  216, 355–373. (ISOFIT)
- Vermote, E.F., El Saleous, N., Justice, C.O., Kaufman, Y.J.,
  Privette, J.L., Remer, L., Roger, J.C. and Tanré, D. (1997):
  Atmospheric correction of visible to middle-infrared EOS-MODIS data
  over land surfaces: Background, operational algorithm and validation.
  *Journal of Geophysical Research: Atmospheres*, 102(D14),
  17131–17141. (adjacency correction)

## AUTHORS

i.hyper.smac project.
