## DESCRIPTION

*i.hyper.atcorr* computes a multi-dimensional atmospheric correction
look-up table (LUT) using a C port of the 6SV2.1 (Second Simulation of
a Satellite Signal in the Solar Spectrum, version 2.1) radiative
transfer code with OpenMP parallelisation over the AOD dimension.

The LUT spans a grid of AOD (aerosol optical depth at 550 nm) and
column water vapour (H₂O) values at a set of user-specified wavelengths.
For each grid point the module stores four atmospheric parameters:

- **R_atm** — atmospheric path reflectance
- **T_down** — total downward transmittance (direct + diffuse)
- **T_up** — total upward transmittance (direct + diffuse)
- **s_alb** — spherical albedo of the atmosphere

These four parameters are sufficient to invert a top-of-atmosphere (TOA)
reflectance image to surface (BOA) reflectance via the standard algebraic
formula:

```
rho_boa = (rho_toa - R_atm) / (T_down * T_up * (1 + s_alb * rho_boa))
```

The output is a compact binary LUT file intended for use by
*i.hyper.smac*, which loads it via the `atcorr.py` Python ctypes
bindings bundled with the add-on.

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
approximately one second.

## EXAMPLES

### Generate a LUT for a Tanager scene over Kanpur

```sh
i.hyper.atcorr \
    output=kanpur_lut.lut \
    sza=35.2 vza=4.1 raa=97 \
    atmosphere=midsum aerosol=continental ozone=310 \
    aod=0.0,0.05,0.1,0.2,0.4,0.8 \
    h2o=1.0,2.0,3.5,5.0 \
    wl_min=0.376 wl_max=2.499 wl_step=0.005
```

### Quick test at coarse spectral resolution

```sh
i.hyper.atcorr --verbose \
    output=/tmp/test.lut \
    sza=30 vza=0 raa=0 \
    aod=0.0,0.2,0.4 h2o=2.0,4.0 \
    wl_min=0.4 wl_max=2.5 wl_step=0.05
```

## SEE ALSO

*[i.hyper.smac](i.hyper.smac.md), [i.atcorr](i.atcorr.md)*

## REFERENCES

- Vermote, E.F., Tanré, D., Deuzé, J.L., Herman, M. and Morcrette, J.J.
  (1997): Second simulation of the satellite signal in the solar
  spectrum, 6S: An overview. *IEEE Transactions on Geoscience and Remote
  Sensing*, 35(3), 675–686.
- Kotchenova, S.Y., Vermote, E.F., Matarrese, R. and Klemm, F.J. (2006):
  Validation of a vector version of the 6S radiative transfer code for
  atmospheric correction of satellite data. *Applied Optics*, 45(26),
  6762–6778.

## AUTHORS

i.hyper.smac project.
