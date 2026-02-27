# i.hyper.atcorr

GRASS GIS add-on for atmospheric correction of hyperspectral imagery using a
C port of the 6SV2.1 (Second Simulation of a Satellite Signal in the Solar
Spectrum) radiative transfer algorithm with OpenMP parallelisation.

---

## Overview

*i.hyper.atcorr* operates in two complementary modes:

| Mode | Trigger | Purpose |
|------|---------|---------|
| **LUT generation** | `lut=` | Compute a binary look-up table of atmospheric parameters over an [AOD × H₂O × wavelength] grid |
| **Cube correction** | `input=` / `output=` | Apply the LUT to a Raster3D radiance cube, writing a surface (BOA) reflectance cube |

Both modes can be combined in a single invocation; the LUT is computed once
and used immediately for correction.

---

## Physics

### Forward model (6SV)

For each (AOD, H₂O, λ) grid point the module stores four atmospheric
parameters:

| Symbol | Name |
|--------|------|
| **R_atm** | Atmospheric path reflectance |
| **T_down** | Total downward transmittance (direct + diffuse) |
| **T_up** | Total upward transmittance (direct + diffuse) |
| **s_alb** | Spherical albedo of the atmosphere |

### Inversion (BOA reflectance)

```
ρ_toa = (π × L × d²) / (E₀ × cos θₛ)
ρ_boa = (ρ_toa − R_atm) / (T_down × T_up × (1 + s_alb × ρ_boa))
```

where *L* is TOA radiance in W/(m² sr µm), *E₀* is the Thuillier solar
irradiance (from 6SV2.1 tables), and *d²* is the squared Earth-Sun distance
for the acquisition day-of-year.

---

## ISOFIT-Inspired Improvements

Six improvements inspired by the
[ISOFIT](https://github.com/isofit/isofit) framework are available as
optional flags and parameters. They can be combined freely; each defaults to
disabled for full backward compatibility.

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
maps before correction, suppressing retrieval noise while preserving large-
scale spatial gradients. Boundary pixels use edge-replication padding;
NaN/null pixels are excluded from the neighbourhood average.

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

After all bands have been inverted, blends each pixel's retrieved spectrum
with a **3-component Gaussian mixture surface prior** (vegetation, soil,
water) using diagonal-covariance MAP estimation:

```
r_map[b] = (r_obs[b] / σ_obs²[b]  +  r_prior[b] / σ_prior²[b])
           / (1/σ_obs²[b]  +  1/σ_prior²[b])
```

Each pixel is classified to the nearest component using VNIR bands
(0.40–1.00 µm). Prior means are hardcoded reference spectra interpolated to
the sensor wavelengths; prior variances scale as `(scale × mean)²`
(vegetation 15 %, soil 20 %, water 3 %). The `-r` flag requires loading the
full reflectance cube into memory (~400 MB for a 426-band Tanager scene).

### #4 — Per-band reflectance uncertainty (`-u` flag)

```
-u   [uncertainty=<output_raster3d>]
```

Computes per-pixel reflectance uncertainty (σ_rfl) for each band from two
propagated sources:

1. **Instrument noise**: NEDL estimated from the standard deviation of the
   darkest 5 % of radiance pixels → propagated as
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
- Gas absorption band edges (720, 760, 940, 1135, 1380, 1850 nm): +2 %
  Gaussian bump (σ = 30 nm)
- SWIR > 1500 nm: +1 % (aerosol model uncertainty)

This prevents over-regularisation in bands where the 6SV gas
parameterisation is least accurate.

---

## Installation

### Prerequisites

- GRASS GIS 8.x compiled from source (or development headers)
- GCC with OpenMP support (`-fopenmp`)
- Set `MODULE_TOPDIR` in `Makefile` to your GRASS source tree

### Build

```sh
cd i.hyper.atcorr
make
```

Output: `$GRASS_DIST/bin/i.hyper.atcorr` and `$GRASS_DIST/lib/libatcorr.so`

### Install to system GRASS

```sh
sudo cp dist.*/bin/i.hyper.atcorr /usr/local/grass85/bin/
sudo cp dist.*/lib/libatcorr.so   /usr/local/grass85/lib/
```

---

## Usage

**Image-based retrieval — which flags to activate per scene type**:

| Scene | Flags | Rationale |
|-------|-------|-----------|
| Saharan dust | `-z dem=` | No DDV (barren desert); dry uniform H₂O; O₃ and elevation matter |
| Amazon / tropical | `-z -w -a` | Dense forest = ideal DDV; ~4 g/cm² WVC gradient; low tropical O₃ |
| Urban temperate winter | `-z -a` | Variable O₃ (polar vortex); farmland DDV; dry winter air |
| Mediterranean coastal | `-w -a` | Land–sea H₂O gradient; coastal DDV; stable O₃ |
| Sub-arctic winter | `-z` | Polar O₃ enhancement; snow = no DDV; very dry |
| Boreal summer / mountain | `-z -w -a dem=` | All four: dense DDV + WVC gradient + variable O₃ + elevation |

### LUT generation only

```sh
i.hyper.atcorr \
    lut=kanpur.lut \
    sza=35.2 vza=4.1 raa=97 \
    atmosphere=midsum aerosol=continental ozone=310 \
    aod=0.0,0.05,0.1,0.2,0.4,0.8 \
    h2o=1.0,2.0,3.5,5.0 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005
```

### Correction with scalar atmospheric state

```sh
i.hyper.atcorr \
    input=tanager_radiance output=tanager_refl \
    lut=kanpur.lut \
    sza=35.2 vza=4.1 raa=97 doy=45 \
    aod_val=0.18 h2o_val=3.5 \
    atmosphere=midsum aerosol=continental
```

### Full ISOFIT pipeline — all 6 improvements

```sh
i.hyper.atcorr -u -r \
    input=tanager_radiance output=tanager_refl \
    sza=35.2 vza=4.1 raa=97 doy=45 \
    aod_val=0.18 h2o_val=3.5 \
    atmosphere=midsum aerosol=continental \
    aod_map=maiac_aod h2o_map=mod05_wvc \
    smooth=3 \
    adj_psf=1.0 \
    uncertainty=tanager_refl_unc
```

Runs with:
- Per-pixel AOD from MAIAC, H₂O from MOD05, Gaussian-smoothed at σ=3 px
- Adjacency correction with 1 km PSF (auto pixel size from region)
- Surface prior MAP regularisation
- Uncertainty output in `tanager_refl_unc`

### Fully standalone — all atmospheric state from the image

No external ancillary products required.  A single command retrieves O₃,
per-pixel H₂O, per-pixel AOD, and surface pressure from the image, then
generates the LUT and applies the correction.

```sh
i.hyper.atcorr -z -w -a \
    input=tanager_radiance output=tanager_refl \
    lut=tanager_auto.lut \
    sza=35.2 vza=4.1 raa=97 doy=45 \
    atmosphere=midsum aerosol=continental \
    dem=srtm_dem \
    aod=0.0,0.05,0.1,0.2,0.4,0.8 \
    h2o=0.5,1.5,3.0,5.0 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005
```

- **`-z`** retrieves scene-mean O₃ (DU) from Chappuis absorption at 600 nm;
  replaces `ozone=` before the LUT is computed
- **`-w`** retrieves per-pixel WVC (g/cm²) from the 940 nm band depth
  (continuum from 865/1040 nm shoulders); used in place of `h2o_val=`
- **`-a`** retrieves per-pixel AOD at 550 nm from the MODIS DDV algorithm
  (470/660/860/2130 nm); non-DDV pixels receive the scene-mean AOD
- **`dem=`** computes ISA surface pressure from the mean DEM elevation and
  updates the LUT configuration before computation

---

## Options Reference

### I/O

| Option | Type | Description |
|--------|------|-------------|
| `input=` | R3 raster | TOA radiance W/(m² sr µm) |
| `output=` | R3 raster | BOA surface reflectance |
| `lut=` | file | Binary LUT output file |

### Geometry

| Option | Default | Description |
|--------|---------|-------------|
| `sza=` | — | Solar zenith angle (°) |
| `vza=` | 0 | View zenith angle (°) |
| `raa=` | 0 | Relative azimuth angle (°) |
| `altitude=` | 1000 | Sensor altitude in km (>900 = satellite) |

### Atmosphere

| Option | Default | Description |
|--------|---------|-------------|
| `atmosphere=` | us62 | Standard atmosphere model |
| `aerosol=` | continental | Aerosol type |
| `ozone=` | 300 | Ozone column (Dobson units) |

### LUT grid

| Option | Default | Description |
|--------|---------|-------------|
| `aod=` | 0.0,0.05,0.1,0.2,0.4,0.8 | AOD at 550 nm grid points |
| `h2o=` | 0.5,1.0,2.0,3.5,5.0 | H₂O grid points (g/cm²) |
| `wl_min=` | 0.40 | Minimum wavelength (µm) |
| `wl_max=` | 2.50 | Maximum wavelength (µm) |
| `wl_step=` | 0.01 | Wavelength step (µm) |

### Correction

| Option | Default | Description |
|--------|---------|-------------|
| `doy=` | 180 | Day of year (Earth-Sun distance) |
| `aod_val=` | 0.1 | Scene-average AOD (scalar fallback) |
| `h2o_val=` | 2.0 | Scene-average H₂O g/cm² (scalar fallback) |

### ISOFIT improvements

| Option / Flag | Default | Description |
|---------------|---------|-------------|
| `aod_map=` | — | Per-pixel AOD raster (#1) |
| `h2o_map=` | — | Per-pixel H₂O raster (#1) |
| `smooth=` | 0 | Gaussian smoothing σ in pixels for atm maps (#1) |
| `adj_psf=` | 0 | Adjacency PSF radius km (0=off) (#2) |
| `pixel_size=` | 0 | Pixel size m (0=auto from region) (#2) |
| `-r` | off | Surface prior MAP regularisation (#3/#5/#6) |
| `-u` | off | Compute per-band uncertainty (#4) |
| `uncertainty=` | — | Output uncertainty Raster3D (#4) |

### Image-based Retrieval

These flags retrieve atmospheric state directly from the input radiance cube,
eliminating the need for co-located ancillary products.  `-z` and `dem=`
update the LUT configuration before it is computed; `-w` and `-a` provide
per-pixel arrays used during the correction step.

| Option / Flag | Description |
|---------------|-------------|
| `-z` | Retrieve scene-mean O₃ (DU) from Chappuis band depth at 600 nm; requires bands near 540, 600, 680 nm |
| `-w` | Retrieve per-pixel WVC (g/cm²) from 940 nm band depth; requires bands near 865, 940, 1040 nm |
| `-a` | Retrieve per-pixel AOD at 550 nm from MODIS DDV algorithm; requires bands near 470, 660, 860, 2130 nm |
| `dem=` | ISA surface pressure from mean elevation of a DEM raster; replaces default 1013.25 hPa |

---

## Architecture

```
src/
├── lut.c            OpenMP LUT computation (AOD outer loop)
│                    + atcorr_lut_interp_pixel() trilinear interp
├── discom.c         6SV scattering at 20 reference wavelengths (SOS)
├── interp.c         Log-log wavelength interpolation
├── gas_abs.c        Curtis-Godson gas transmittance
├── aerosol.c        Aerosol mixture initialisation
├── atmosphere.c     Standard atmosphere models
├── srf_conv.c       SRF gas correction via libRadtran reptran fine
├── spatial.c        Separable Gaussian + box filters          [#1]
├── adjacency.c      Vermote 1997 adjacency correction         [#2]
├── surface_model.c  3-component surface prior + MAP           [#3,#5,#6]
├── uncertainty.c    Noise + AOD-perturbation uncertainty      [#4]
└── rt.c / scatra.c / ... (6SV RT solver, ported from Fortran)

include/
├── atcorr.h         Public API (LutConfig, LutArrays, all exports)
├── spatial.h        [#1]
├── adjacency.h      [#2]
├── surface_model.h  [#3,#5,#6]
└── uncertainty.h    [#4]

main.c               GRASS module interface, correct_raster3d()

python/
└── atcorr.py        ctypes bindings (LutConfig, compute_lut, lut_slice,
                     atcorr_lut_interp_pixel, apply_srf_correction)
```

---

## LUT file format

Host-endian binary (little-endian on x86):

```
magic     uint32   0x4C555400  ("LUT\0")
version   uint32   1
n_aod     int32
n_h2o     int32
n_wl      int32
aod[n_aod]                     float32
h2o[n_h2o]                     float32
wl [n_wl]                      float32  (micrometres)
R_atm [n_aod × n_h2o × n_wl]  float32
T_down[n_aod × n_h2o × n_wl]  float32
T_up  [n_aod × n_h2o × n_wl]  float32
s_alb [n_aod × n_h2o × n_wl]  float32
```

C order: wavelength index varies fastest.
Typical size: ~4 MB for 6 AOD × 5 H₂O × 211 wavelengths.

---

## Python bindings

`libatcorr.so` is importable from Python via ctypes (`python/atcorr.py`):

```python
from atcorr import LutConfig, compute_lut, lut_slice, apply_srf_correction

cfg = LutConfig(sza=35.2, vza=4.1, raa=97, ...)
lut = compute_lut(cfg)          # returns LutArrays with numpy arrays
Rs, Tds, Tus, ss = lut_slice(cfg, lut, aod_val=0.18, h2o_val=3.5)
lut = apply_srf_correction(cfg, lut, fwhm_um=band_fwhm, threshold_nm=5.0)
```

---

## Validation — Fortran 6SV2.1 compatibility

`testsuite/test_fortran_compat.py` (25 tests) cross-checks every C function
in `libatcorr.so` against the original Fortran 77 subroutines compiled from
`~/dev/6SV2.1/`.  All 25 tests pass with no C code bugs found.

| Subroutine | Function tested | Tests | Tolerance | Actual agreement |
|---|---|---|---|---|
| CHAND | `sixs_chand()` – Chandrasekhar Rayleigh reflectance | 4 geometries | rtol=1×10⁻⁵ | ~7×10⁻⁸ (float32 limit) |
| ODRAYL | `sixs_odrayl()` – Rayleigh optical depth (Edlén 1966) | 4 wavelengths | rtol=5×10⁻³ | ~2×10⁻⁷ |
| VARSOL × d² ≈ 1 | `sixs_earth_sun_dist2()` – Earth-Sun distance | 4 DOYs | rtol=5×10⁻³ | <0.07% |
| SOLIRR / E0 on-grid | `sixs_E0()` – Thuillier solar irradiance | 4 wavelengths | rtol=1×10⁻³ | 0–3 ULP |
| SOLIRR / E0 off-grid | `sixs_E0()` – linear vs nearest-neighbour interp | 1 wavelength | rtol=1×10⁻³ | 0.035% |
| CSALBR | `sixs_csalbr()` – Rayleigh spherical albedo | 3 τ values | rtol=1×10⁻⁵ | ~9×10⁻⁸ (float32 limit) |
| GAUSS | `sixs_gauss()` – Gauss-Legendre quadrature | n=4 and n=8, weight sums, symmetry | rtol=1×10⁻⁵ | Exact float32 |

Minor intentional differences:
- `sixs_E0()` uses linear interpolation for off-grid wavelengths; Fortran SOLIRR uses nearest-neighbour — both agree to 0.035% (smooth solar spectrum).
- `sixs_earth_sun_dist2()` returns d² (< 1 at perihelion); Fortran VARSOL returns 1/d² (> 1 at perihelion) — the product ≈ 1.0 within 0.07%, confirming complementary conventions.
- Solar table float32 literals produce 0–3 ULP differences between gfortran and the C compiler at parse time — not a bug.

**Build and run:**

```sh
# Compile 6SV2.1 Fortran objects (first time only)
cd ~/dev/6sV2.1
gfortran -O -ffixed-line-length-132 -c CHAND.f ODRAYL.f VARSOL.f SOLIRR.f CSALBR.f GAUSS.f US62.f

# Run all 25 compatibility tests
cd ~/dev/i.hyper.atcorr
grass --tmp-project XY --exec python3 testsuite/test_fortran_compat.py
```

The test driver (`testsuite/test_6sv_compat.f90`) is compiled automatically
by the Python test suite when the Fortran objects are present.

---

## References

- Vermote, E.F., Tanré, D., Deuzé, J.L., Herman, M. and Morcrette, J.J.
  (1997): Second simulation of the satellite signal in the solar spectrum,
  6S: An overview. *IEEE Trans. Geosci. Remote Sens.*, 35(3), 675–686.
- Kotchenova, S.Y., Vermote, E.F., Matarrese, R. and Klemm, F.J. (2006):
  Validation of a vector version of the 6S radiative transfer code for
  atmospheric correction of satellite data. *Applied Optics*, 45(26),
  6762–6778.
- Thompson, D.R. et al. (2018): Optimal estimation for imaging
  spectrometer atmospheric correction. *Remote Sensing of Environment*,
  216, 355–373. (ISOFIT)
- Vermote, E.F., El Saleous, N., Justice, C.O., Kaufman, Y.J.,
  Privette, J.L., Remer, L., Roger, J.C. and Tanré, D. (1997):
  Atmospheric correction of visible to middle-infrared EOS-MODIS data over
  land surfaces: Background, operational algorithm and validation.
  *J. Geophys. Res. Atmos.*, 102(D14), 17131–17141. (adjacency correction)

## Authors

i.hyper.smac project.
