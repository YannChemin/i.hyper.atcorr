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
- Adjacency correction with 1 km PSF (pixel size auto-detected from region)
- Surface prior MAP regularisation (vegetation / soil / water prior)
- Uncertainty output in `tanager_refl_unc` Raster3D

### Quick test at coarse spectral resolution

```sh
i.hyper.atcorr --verbose \
    lut=/tmp/test.lut \
    sza=30 vza=0 raa=0 \
    aod=0.0,0.2,0.4 h2o=2.0,4.0 \
    wl_min=0.4 wl_max=2.5 wl_step=0.05
```

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
