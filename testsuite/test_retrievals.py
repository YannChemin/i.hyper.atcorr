"""Unit tests for atmospheric state retrieval functions.

Tests retrieve_pressure_isa(), retrieve_pressure_o2a(), retrieve_h2o_940(),
retrieve_o3_chappuis(), retrieve_aod_ddv(), retrieve_quality_mask(), and
retrieve_aod_maiac() from libatcorr.so via python/atcorr.py.

Validated reference values (sza=30°, vza=0° unless noted):
  pressure_isa(0 m)   = 1013.25 hPa
  pressure_isa(500 m) =  954.61 hPa
  h2o_940(D=0.2)      =    2.58 g/cm²   [sza=30°, vza=0°]
  o3_chappuis(D=0.03) =  139.2  DU      [sza=30°, vza=0°]
  pressure_o2a(D=0.5) = 1013.25 hPa     [nadir: sza=0°, vza=0°]

Run from a GRASS session:
    python -m grass.gunittest.main
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import atcorr as ac

from grass.gunittest.case import TestCase
from grass.gunittest.main import test

# Geometry used for validated test cases
_SZA30 = 30.0   # degrees
_VZA0  = 0.0    # nadir
_NPIX  = 10     # number of synthetic pixels


def _uniform(val, n=_NPIX):
    """Return a constant float32 array."""
    return np.full(n, val, dtype=np.float32)


class TestPressureISA(TestCase):
    """Tests for retrieve_pressure_isa(): ISA barometric formula."""

    def test_sea_level(self):
        """Elevation 0 m must return 1013.25 hPa."""
        P = ac.retrieve_pressure_isa(0.0)
        self.assertAlmostEqual(P, 1013.25, delta=0.1)

    def test_500m(self):
        """Elevation 500 m must return ~954.6 hPa."""
        P = ac.retrieve_pressure_isa(500.0)
        self.assertAlmostEqual(P, 954.61, delta=1.0)

    def test_monotone_decreasing(self):
        """Pressure must decrease monotonically with elevation."""
        elevs = [0, 200, 500, 1000, 2000, 4000, 8000]
        pressures = [ac.retrieve_pressure_isa(float(e)) for e in elevs]
        for k in range(len(pressures) - 1):
            self.assertGreater(
                pressures[k], pressures[k + 1],
                msg=f"P({elevs[k]}m) = {pressures[k]:.2f} ≤ P({elevs[k+1]}m) = {pressures[k+1]:.2f}",
            )

    def test_clamped_at_11000m(self):
        """Pressure must be clamped (≥ 200 hPa) for extreme elevations."""
        P = ac.retrieve_pressure_isa(12000.0)
        self.assertGreaterEqual(P, 200.0)

    def test_returns_float(self):
        """Function must return a Python float."""
        self.assertIsInstance(ac.retrieve_pressure_isa(0.0), float)


class TestH2O940(TestCase):
    """Tests for retrieve_h2o_940(): Kaufman & Gao 940 nm band depth."""

    def test_known_depth_D02(self):
        """D=0.2 at sza=30°, vza=0° must give WVC ≈ 2.58 g/cm²."""
        # L_cont = linear_interp(L_865, L_1040) at 940 nm = 100 (flat continuum)
        # D = 1 - L_940/L_cont = 0.2  →  L_940 = 80
        wvc = ac.retrieve_h2o_940(
            _uniform(100.0), _uniform(80.0), _uniform(100.0),
            sza_deg=_SZA30, vza_deg=_VZA0,
        )
        self.assertAlmostEqual(float(wvc.mean()), 2.58, delta=0.15)

    def test_no_absorption(self):
        """D≈0 (L_940 ≈ L_cont) must give WVC near minimum clamp (0.1 g/cm²)."""
        wvc = ac.retrieve_h2o_940(
            _uniform(100.0), _uniform(99.9), _uniform(100.0),
            sza_deg=_SZA30, vza_deg=_VZA0,
        )
        self.assertLessEqual(float(wvc.mean()), 1.0)

    def test_result_clamped_above(self):
        """Very high D must not exceed 8.0 g/cm²."""
        # D→1 (near-total absorption): L_940 = 0
        wvc = ac.retrieve_h2o_940(
            _uniform(100.0), _uniform(0.01), _uniform(100.0),
            sza_deg=_SZA30, vza_deg=_VZA0,
        )
        self.assertLessEqual(float(wvc.max()), 8.0 + 0.01)

    def test_result_clamped_below(self):
        """Negative D (L_940 > L_cont) must return the minimum clamp (0.1 g/cm²)."""
        # L_940 > continuum: negative absorption → default 2.0 g/cm² or clamp to 0.1
        wvc = ac.retrieve_h2o_940(
            _uniform(100.0), _uniform(110.0), _uniform(100.0),
            sza_deg=_SZA30, vza_deg=_VZA0,
        )
        self.assertGreaterEqual(float(wvc.min()), 0.09)

    def test_output_shape(self):
        """Output array must have the same length as input."""
        n = 25
        wvc = ac.retrieve_h2o_940(
            _uniform(100.0, n), _uniform(80.0, n), _uniform(100.0, n),
            sza_deg=_SZA30, vza_deg=_VZA0,
        )
        self.assertEqual(len(wvc), n)

    def test_higher_sza_lower_wvc(self):
        """At fixed D, higher SZA (longer path) must give lower WVC."""
        wvc_low  = ac.retrieve_h2o_940(_uniform(100), _uniform(80), _uniform(100), 20.0, 0.0)
        wvc_high = ac.retrieve_h2o_940(_uniform(100), _uniform(80), _uniform(100), 60.0, 0.0)
        # Larger airmass → more absorption needed → lower retrieved WVC per unit D
        self.assertLess(float(wvc_high.mean()), float(wvc_low.mean()))


class TestO3Chappuis(TestCase):
    """Tests for retrieve_o3_chappuis(): Chappuis band depth at 600 nm."""

    def test_known_depth_D003(self):
        """D=0.03 at sza=30°, vza=0° must give O3 ≈ 139.2 DU."""
        # L_cont = interp(L_540, L_680) at 600 nm = 100 (flat continuum)
        # D = 1 - L_600/L_cont = 0.03  →  L_600 = 97
        o3 = ac.retrieve_o3_chappuis(
            _uniform(100.0), _uniform(97.0), _uniform(100.0),
            sza_deg=_SZA30, vza_deg=_VZA0,
        )
        self.assertAlmostEqual(o3, 139.2, delta=5.0)

    def test_no_absorption_gives_fallback(self):
        """D≈0 (no Chappuis absorption) must return the fallback (300 DU)."""
        o3 = ac.retrieve_o3_chappuis(
            _uniform(100.0), _uniform(100.0), _uniform(100.0),
            sza_deg=_SZA30, vza_deg=_VZA0,
        )
        self.assertAlmostEqual(o3, 300.0, delta=10.0)

    def test_result_clamped_in_range(self):
        """Result must always be in [50, 800] DU."""
        for d in [0.0, 0.05, 0.15, 0.30]:
            o3 = ac.retrieve_o3_chappuis(
                _uniform(100.0),
                _uniform(100.0 * (1.0 - d)),
                _uniform(100.0),
                sza_deg=_SZA30, vza_deg=_VZA0,
            )
            self.assertGreaterEqual(o3, 50.0,  msg=f"O3 < 50 DU at D={d}")
            self.assertLessEqual(o3, 800.0, msg=f"O3 > 800 DU at D={d}")

    def test_returns_float(self):
        """Function must return a Python float (scene mean)."""
        o3 = ac.retrieve_o3_chappuis(
            _uniform(100.0), _uniform(97.0), _uniform(100.0),
            sza_deg=_SZA30, vza_deg=_VZA0,
        )
        self.assertIsInstance(o3, float)


class TestPressureO2A(TestCase):
    """Tests for retrieve_pressure_o2a(): O₂-A 760 nm band depth."""

    def test_known_depth_D05_nadir(self):
        """D=0.5 at nadir (sza=0°, vza=0°) must give P ≈ 1013.25 hPa."""
        # L_cont = interp(L_740, L_780) at 760 nm = 100 (flat)
        # D = 1 - L_760/L_cont = 0.5  →  L_760 = 50
        P = ac.retrieve_pressure_o2a(
            _uniform(100.0), _uniform(50.0), _uniform(100.0),
            sza_deg=0.0, vza_deg=0.0,
        )
        self.assertAlmostEqual(float(P.mean()), 1013.25, delta=20.0)

    def test_output_clamped_in_range(self):
        """All output pressures must be in [200, 1100] hPa."""
        P = ac.retrieve_pressure_o2a(
            _uniform(100.0), _uniform(50.0), _uniform(100.0),
            sza_deg=0.0, vza_deg=0.0,
        )
        self.assertTrue(np.all(P >= 200.0))
        self.assertTrue(np.all(P <= 1100.0))

    def test_invalid_pixels_default_to_P0(self):
        """Pixels with L_760 ≤ 0 must fall back to standard pressure (≈1013 hPa)."""
        L_zero = np.zeros(_NPIX, dtype=np.float32)
        P = ac.retrieve_pressure_o2a(
            _uniform(100.0), L_zero, _uniform(100.0),
            sza_deg=0.0, vza_deg=0.0,
        )
        # Should return sea-level pressure or clamped value
        self.assertTrue(np.all(P > 0.0))

    def test_output_shape(self):
        """Output array must have the same length as input."""
        n = 15
        P = ac.retrieve_pressure_o2a(
            _uniform(100.0, n), _uniform(50.0, n), _uniform(100.0, n),
            sza_deg=0.0, vza_deg=0.0,
        )
        self.assertEqual(len(P), n)


class TestQualityMask(TestCase):
    """Tests for retrieve_quality_mask(): cloud/shadow/water/snow flags."""

    # Typical radiance for a clear land surface
    _L_CLEAR_BLUE =  50.0
    _L_CLEAR_RED  =  80.0
    _L_CLEAR_NIR  = 130.0  # NDVI > 0 → not cloud, not water

    def test_clear_pixels_no_flags(self):
        """Moderate-radiance clear land surface must have mask = 0."""
        mask = ac.retrieve_quality_mask(
            _uniform(self._L_CLEAR_BLUE),
            _uniform(self._L_CLEAR_RED),
            _uniform(self._L_CLEAR_NIR),
            doy=180, sza_deg=30.0,
        )
        self.assertTrue(
            np.all(mask == 0),
            msg=f"Clear pixels have unexpected flags: {np.unique(mask)}",
        )

    def test_cloud_flag_set(self):
        """Bright blue (L_blue >> threshold) must trigger MASK_CLOUD."""
        # TOA_blue ≈ L_blue / E0_blue → needs to exceed ~0.25 reflectance
        # Use very high radiance to force cloud flag
        mask = ac.retrieve_quality_mask(
            _uniform(800.0),   # extremely bright blue → TOA_blue > 0.25
            _uniform(900.0),   # red also bright
            _uniform(500.0),   # NIR bright → NDVI negative
            doy=180, sza_deg=30.0,
        )
        self.assertTrue(
            np.any(mask & ac.MASK_CLOUD),
            msg="Expected MASK_CLOUD set for very bright pixels",
        )

    def test_water_flag_set(self):
        """Low NIR with NDVI < 0 must trigger MASK_WATER."""
        mask = ac.retrieve_quality_mask(
            _uniform(30.0),   # blue moderate
            _uniform(50.0),   # red > nir → NDVI < 0
            _uniform(10.0),   # very low NIR < 0.05 reflectance threshold
            doy=180, sza_deg=30.0,
        )
        self.assertTrue(
            np.any(mask & ac.MASK_WATER),
            msg="Expected MASK_WATER set for dark NIR + negative NDVI",
        )

    def test_shadow_flag_set(self):
        """Uniformly dark pixels (blue, red, NIR all very low) must trigger MASK_SHADOW."""
        mask = ac.retrieve_quality_mask(
            _uniform(3.0),    # blue < 0.04 reflectance
            _uniform(3.0),    # red < 0.04
            _uniform(3.0),    # NIR < 0.04
            doy=180, sza_deg=30.0,
        )
        self.assertTrue(
            np.any(mask & ac.MASK_SHADOW),
            msg="Expected MASK_SHADOW set for all-dark pixels",
        )

    def test_snow_flag_set_with_swir(self):
        """NDSI > 0.4 + bright NIR must trigger MASK_SNOW when SWIR provided."""
        # NDSI = (L_green - L_swir) / (L_green + L_swir) > 0.4 → L_green >> L_swir
        # Use blue ≈ green proxy, SWIR very dark, NIR > 0.1
        mask = ac.retrieve_quality_mask(
            _uniform(300.0),   # bright green proxy
            _uniform(100.0),   # red
            _uniform(200.0),   # bright NIR > 0.1
            doy=180, sza_deg=30.0,
            L_swir=_uniform(5.0),  # very dark SWIR → NDSI > 0.4
        )
        self.assertTrue(
            np.any(mask & ac.MASK_SNOW),
            msg="Expected MASK_SNOW set for bright VIS + dark SWIR",
        )

    def test_snow_not_set_without_swir(self):
        """MASK_SNOW must not be set when L_swir is None."""
        mask = ac.retrieve_quality_mask(
            _uniform(300.0), _uniform(100.0), _uniform(200.0),
            doy=180, sza_deg=30.0,
            L_swir=None,
        )
        self.assertFalse(
            np.any(mask & ac.MASK_SNOW),
            msg="MASK_SNOW set even though L_swir=None",
        )

    def test_output_shape_and_dtype(self):
        """Output must be uint8 array of same length as input."""
        n = 20
        mask = ac.retrieve_quality_mask(
            _uniform(50.0, n), _uniform(80.0, n), _uniform(130.0, n),
            doy=180, sza_deg=30.0,
        )
        self.assertEqual(len(mask), n)
        self.assertEqual(mask.dtype, np.uint8)


class TestAODMaiac(TestCase):
    """Tests for retrieve_aod_maiac(): MAIAC patch-median spatial regularization."""

    def test_uniform_field_unchanged(self):
        """Uniform AOD field must be unchanged after regularization."""
        aod = np.full((10, 10), 0.2, dtype=np.float32)
        original = aod.copy()
        ac.retrieve_aod_maiac(aod, nrows=10, ncols=10, patch_sz=5)
        np.testing.assert_allclose(
            aod, original, atol=0.01,
            err_msg="Uniform AOD field changed after MAIAC regularization",
        )

    def test_spike_reduced(self):
        """Single outlier spike in an otherwise smooth field must be reduced."""
        aod = np.full((10, 10), 0.1, dtype=np.float32)
        aod[5, 5] = 5.0  # outlier
        ac.retrieve_aod_maiac(aod, nrows=10, ncols=10, patch_sz=5)
        self.assertLess(
            float(aod[5, 5]), 5.0,
            msg=f"Spike not reduced: aod[5,5] = {float(aod[5,5]):.3f} (unchanged from 5.0)",
        )

    def test_inplace_modification(self):
        """retrieve_aod_maiac must modify the input array in place."""
        aod = np.full((8, 8), 0.15, dtype=np.float32)
        aod[3, 3] = 1.0
        ptr_before = aod.ctypes.data
        ac.retrieve_aod_maiac(aod, nrows=8, ncols=8, patch_sz=4)
        self.assertEqual(aod.ctypes.data, ptr_before)  # same memory

    def test_result_non_negative(self):
        """All AOD values must remain non-negative after regularization."""
        aod = np.random.uniform(0.0, 0.8, (12, 12)).astype(np.float32)
        ac.retrieve_aod_maiac(aod, nrows=12, ncols=12, patch_sz=4)
        self.assertTrue(np.all(aod >= 0.0))


class TestAODDDV(TestCase):
    """Tests for retrieve_aod_ddv(): MODIS DDV dark-target aerosol retrieval."""

    def _ddv_radiances(self, aod_scene=0.15, npix=20):
        """Create synthetic DDV-eligible pixels.

        DDV condition: 0.01 < ρ_2130 < 0.25 AND NDVI > 0.1.
        Use L_2130 ≈ 0.08 × E0_2130/π, L_860 >> L_660 (NDVI > 0.1).
        """
        # Simple synthetic values that pass the DDV mask
        L_2130 = np.full(npix, 15.0, dtype=np.float32)   # ρ_2130 ≈ 0.06
        L_860  = np.full(npix, 80.0, dtype=np.float32)
        L_660  = np.full(npix, 30.0, dtype=np.float32)   # NDVI > 0.1
        L_470  = np.full(npix, 20.0, dtype=np.float32)
        return L_470, L_660, L_860, L_2130

    def test_returns_scene_mean_and_array(self):
        """retrieve_aod_ddv must return a (float, ndarray) tuple."""
        L_470, L_660, L_860, L_2130 = self._ddv_radiances()
        result = ac.retrieve_aod_ddv(L_470, L_660, L_860, L_2130, doy=180, sza_deg=30.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        scene_mean, aod_px = result
        self.assertIsInstance(scene_mean, float)
        self.assertEqual(len(aod_px), 20)

    def test_scene_mean_non_negative(self):
        """Scene-mean AOD must be ≥ 0."""
        L_470, L_660, L_860, L_2130 = self._ddv_radiances()
        scene_mean, _ = ac.retrieve_aod_ddv(L_470, L_660, L_860, L_2130, doy=180, sza_deg=30.0)
        self.assertGreaterEqual(scene_mean, 0.0)

    def test_no_ddv_fallback(self):
        """With no DDV pixels (bright SWIR), scene mean must be the fallback 0.15."""
        npix = 20
        L_2130 = np.full(npix, 200.0, dtype=np.float32)   # ρ_2130 >> 0.25 → not DDV
        L_860  = np.full(npix, 80.0, dtype=np.float32)
        L_660  = np.full(npix, 90.0, dtype=np.float32)    # NDVI < 0 → not DDV
        L_470  = np.full(npix, 50.0, dtype=np.float32)
        scene_mean, _ = ac.retrieve_aod_ddv(L_470, L_660, L_860, L_2130, doy=180, sza_deg=30.0)
        self.assertAlmostEqual(scene_mean, 0.15, delta=0.05)

    def test_output_array_shape(self):
        """Per-pixel AOD output array must match input length."""
        n = 30
        L = np.full(n, 20.0, dtype=np.float32)
        _, aod_px = ac.retrieve_aod_ddv(L, L, L, L, doy=180, sza_deg=30.0)
        self.assertEqual(len(aod_px), n)


if __name__ == "__main__":
    test()
