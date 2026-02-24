"""Python ctypes bindings for libatcorr.so (6SV2.1-based atmospheric correction).

Usage:
    from atcorr import LutConfig, LutArrays, compute_lut, lut_slice, invert, solar_E0, earth_sun_dist2

    cfg = LutConfig(
        wl=np.array([0.45, 0.55, 0.65, 0.85], dtype=np.float32),
        aod=np.array([0.05, 0.1, 0.2, 0.4], dtype=np.float32),
        h2o=np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32),
        sza=30.0, vza=5.0, raa=90.0,
        altitude_km=1000.0,   # satellite
        atmo_model=1,         # US62
        aerosol_model=1,      # continental
        surface_pressure=0.0,
        ozone_du=300.0,
    )
    lut = compute_lut(cfg)
    # lut.R_atm, lut.T_down, lut.T_up, lut.s_alb are numpy arrays [n_aod, n_h2o, n_wl]
"""

import ctypes
import os
import numpy as np

# ── Load shared library ───────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_gisbase = os.environ.get("GISBASE", "")
_lib_candidates = [
    # 1. Installed as a GRASS add-on (preferred)
    os.path.join(_gisbase, "lib", "libatcorr.so"),
    # 2. Same directory as this script (installed alongside)
    os.path.join(_here, "libatcorr.so"),
    # 3. Build directory (developer build)
    os.path.join(_here, "..", "build", "libatcorr.so"),
    os.path.join(_here, "..", "..", "i.hyper.atcorr", "build", "libatcorr.so"),
]
_lib = None
for _path in _lib_candidates:
    if os.path.exists(_path):
        _lib = ctypes.CDLL(_path)
        break
if _lib is None:
    raise ImportError(
        f"libatcorr.so not found. Build with 'make' in i.hyper.atcorr/ first.\n"
        f"Searched: {_lib_candidates}"
    )


# ── C struct mirrors ──────────────────────────────────────────────────────────
class _LutConfigC(ctypes.Structure):
    _fields_ = [
        ("wl",               ctypes.POINTER(ctypes.c_float)),
        ("n_wl",             ctypes.c_int),
        ("aod",              ctypes.POINTER(ctypes.c_float)),
        ("n_aod",            ctypes.c_int),
        ("h2o",              ctypes.POINTER(ctypes.c_float)),
        ("n_h2o",            ctypes.c_int),
        ("sza",              ctypes.c_float),
        ("vza",              ctypes.c_float),
        ("raa",              ctypes.c_float),
        ("altitude_km",      ctypes.c_float),
        ("atmo_model",       ctypes.c_int),
        ("aerosol_model",    ctypes.c_int),
        ("surface_pressure", ctypes.c_float),
        ("ozone_du",         ctypes.c_float),
    ]


class _LutArraysC(ctypes.Structure):
    _fields_ = [
        ("R_atm",  ctypes.POINTER(ctypes.c_float)),
        ("T_down", ctypes.POINTER(ctypes.c_float)),
        ("T_up",   ctypes.POINTER(ctypes.c_float)),
        ("s_alb",  ctypes.POINTER(ctypes.c_float)),
    ]


# ── Function signatures ───────────────────────────────────────────────────────
_lib.atcorr_compute_lut.argtypes = [
    ctypes.POINTER(_LutConfigC),
    ctypes.POINTER(_LutArraysC),
]
_lib.atcorr_compute_lut.restype = ctypes.c_int

_lib.atcorr_version.argtypes = []
_lib.atcorr_version.restype  = ctypes.c_char_p

_lib.sixs_E0.argtypes = [ctypes.c_float]
_lib.sixs_E0.restype  = ctypes.c_float

_lib.sixs_earth_sun_dist2.argtypes = [ctypes.c_int]
_lib.sixs_earth_sun_dist2.restype  = ctypes.c_double

_lib.atcorr_lut_slice.argtypes = [
    ctypes.POINTER(_LutConfigC),
    ctypes.POINTER(_LutArraysC),
    ctypes.c_float,                          # aod_val
    ctypes.c_float,                          # h2o_val
    ctypes.POINTER(ctypes.c_float),          # Rs  [n_wl]
    ctypes.POINTER(ctypes.c_float),          # Tds [n_wl]
    ctypes.POINTER(ctypes.c_float),          # Tus [n_wl]
    ctypes.POINTER(ctypes.c_float),          # ss  [n_wl]
]
_lib.atcorr_lut_slice.restype = None


def version() -> str:
    return _lib.atcorr_version().decode()


def solar_E0(wl_um):
    """Solar irradiance E0 in W/(m² µm) from 6SV2.1 Thuillier spectrum.

    Parameters
    ----------
    wl_um : float or array-like
        Wavelength in µm.
    """
    wl_arr = np.asarray(wl_um, dtype=np.float32)
    scalar = wl_arr.ndim == 0
    wl_arr = np.atleast_1d(wl_arr)
    out = np.array([_lib.sixs_E0(float(w)) for w in wl_arr], dtype=np.float32)
    return float(out[0]) if scalar else out


def earth_sun_dist2(doy):
    """Squared Earth-Sun distance d² in AU for day-of-year (1–365)."""
    return _lib.sixs_earth_sun_dist2(int(doy))


# ── Public Python API ─────────────────────────────────────────────────────────
class LutConfig:
    """Configuration for LUT computation.

    Parameters
    ----------
    wl            : 1D array of wavelengths (µm), float32
    aod           : 1D array of AOD at 550 nm values, float32
    h2o           : 1D array of column water vapour (g/cm²), float32
    sza           : solar zenith angle (degrees)
    vza           : view zenith angle (degrees)
    raa           : relative azimuth angle (degrees)
    altitude_km   : sensor altitude in km (>900 = satellite)
    atmo_model    : atmosphere model (1=US62, 2=MIDSUM, ...)
    aerosol_model : aerosol model (0=none, 1=continental, 2=maritime, 3=urban)
    surface_pressure : surface pressure hPa (0 = use standard atmosphere)
    ozone_du      : ozone column Dobson units (0 = standard atmosphere)
    """
    def __init__(self, wl, aod, h2o,
                 sza=30.0, vza=5.0, raa=90.0,
                 altitude_km=1000.0,
                 atmo_model=1, aerosol_model=1,
                 surface_pressure=0.0, ozone_du=300.0):
        self.wl   = np.asarray(wl,  dtype=np.float32)
        self.aod  = np.asarray(aod, dtype=np.float32)
        self.h2o  = np.asarray(h2o, dtype=np.float32)
        self.sza  = float(sza)
        self.vza  = float(vza)
        self.raa  = float(raa)
        self.altitude_km      = float(altitude_km)
        self.atmo_model       = int(atmo_model)
        self.aerosol_model    = int(aerosol_model)
        self.surface_pressure = float(surface_pressure)
        self.ozone_du         = float(ozone_du)


class LutArrays:
    """Output from compute_lut: numpy arrays shaped [n_aod, n_h2o, n_wl]."""
    def __init__(self, R_atm, T_down, T_up, s_alb):
        self.R_atm  = R_atm
        self.T_down = T_down
        self.T_up   = T_up
        self.s_alb  = s_alb


def compute_lut(cfg: LutConfig) -> LutArrays:
    """Compute a full atmospheric correction LUT.

    Returns LutArrays with R_atm, T_down, T_up, s_alb shaped [n_aod, n_h2o, n_wl].
    """
    n_aod = len(cfg.aod)
    n_h2o = len(cfg.h2o)
    n_wl  = len(cfg.wl)
    n     = n_aod * n_h2o * n_wl

    # Allocate output arrays
    R_atm  = np.zeros(n, dtype=np.float32)
    T_down = np.ones(n,  dtype=np.float32)
    T_up   = np.ones(n,  dtype=np.float32)
    s_alb  = np.zeros(n, dtype=np.float32)

    # Build C structs
    c_cfg = _LutConfigC(
        wl               = cfg.wl.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_wl             = n_wl,
        aod              = cfg.aod.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_aod            = n_aod,
        h2o              = cfg.h2o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_h2o            = n_h2o,
        sza              = cfg.sza,
        vza              = cfg.vza,
        raa              = cfg.raa,
        altitude_km      = cfg.altitude_km,
        atmo_model       = cfg.atmo_model,
        aerosol_model    = cfg.aerosol_model,
        surface_pressure = cfg.surface_pressure,
        ozone_du         = cfg.ozone_du,
    )
    c_out = _LutArraysC(
        R_atm  = R_atm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        T_down = T_down.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        T_up   = T_up.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        s_alb  = s_alb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    ret = _lib.atcorr_compute_lut(ctypes.byref(c_cfg), ctypes.byref(c_out))
    if ret != 0:
        raise RuntimeError(f"atcorr_compute_lut failed with code {ret}")

    shape = (n_aod, n_h2o, n_wl)
    return LutArrays(
        R_atm  = R_atm.reshape(shape),
        T_down = T_down.reshape(shape),
        T_up   = T_up.reshape(shape),
        s_alb  = s_alb.reshape(shape),
    )


def lut_slice(cfg: LutConfig, arrays: LutArrays,
              aod_val: float, h2o_val: float) -> LutArrays:
    """Bilinear interpolation of the LUT at a fixed (aod_val, h2o_val).

    Returns a LutArrays object whose four arrays are 1-D [n_wl] spectral
    curves at the requested atmospheric state.  Useful for pre-computing
    scene-average correction parameters before pixel-level inversion.
    """
    n_wl = len(cfg.wl)
    Rs   = np.zeros(n_wl, dtype=np.float32)
    Tds  = np.zeros(n_wl, dtype=np.float32)
    Tus  = np.zeros(n_wl, dtype=np.float32)
    ss   = np.zeros(n_wl, dtype=np.float32)

    n_aod = len(cfg.aod)
    n_h2o = len(cfg.h2o)
    n     = n_aod * n_h2o * n_wl

    # Flatten LutArrays to 1-D for the C call
    Ra  = arrays.R_atm.ravel().astype(np.float32)
    Td  = arrays.T_down.ravel().astype(np.float32)
    Tu  = arrays.T_up.ravel().astype(np.float32)
    sa  = arrays.s_alb.ravel().astype(np.float32)
    assert len(Ra) == n

    c_cfg = _LutConfigC(
        wl=cfg.wl.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_wl=n_wl,
        aod=cfg.aod.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_aod=n_aod,
        h2o=cfg.h2o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_h2o=n_h2o,
        sza=cfg.sza, vza=cfg.vza, raa=cfg.raa,
        altitude_km=cfg.altitude_km,
        atmo_model=cfg.atmo_model, aerosol_model=cfg.aerosol_model,
        surface_pressure=cfg.surface_pressure, ozone_du=cfg.ozone_du,
    )
    c_arr = _LutArraysC(
        R_atm =Ra.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        T_down=Td.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        T_up  =Tu.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        s_alb =sa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    _lib.atcorr_lut_slice(
        ctypes.byref(c_cfg), ctypes.byref(c_arr),
        ctypes.c_float(aod_val), ctypes.c_float(h2o_val),
        Rs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Tds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Tus.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ss.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return LutArrays(Rs, Tds, Tus, ss)


def invert(rho_toa, R_atm, T_down, T_up, s_alb):
    """Invert TOA reflectance to surface (BOA) reflectance.

    All arrays must be broadcastable. Returns surface reflectance.
    """
    y = (rho_toa - R_atm) / (T_down * T_up + 1e-10)
    return y / (1.0 + s_alb * y + 1e-10)
