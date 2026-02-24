/* Solar irradiance (E0) and Earth-Sun distance from 6SV2.1 data.
 * These functions are part of the public library so Python ctypes bindings
 * can call them independently. */
#define _GNU_SOURCE
#include "solar_table.h"
#include <math.h>

/* Returns solar irradiance E0 in W/(m² µm) at wavelength wl_um.
 * Source: 6SV2.1 Thuillier 2003 spectrum, 1501 points 0.25–4.0 µm at 0.0025
 * µm step.  Linear interpolation; clamped at boundaries. */
float sixs_E0(float wl_um)
{
    float t = (wl_um - (float)SOLAR_TABLE_WL_START) / (float)SOLAR_TABLE_STEP;
    int   i = (int)t;
    float f = t - (float)i;

    if (i < 0)                  return solar_si[0];
    if (i >= SOLAR_TABLE_N - 1) return solar_si[SOLAR_TABLE_N - 1];
    return solar_si[i] * (1.0f - f) + solar_si[i + 1] * f;
}

/* Returns d² (squared Earth-Sun distance in AU) for day-of-year doy (1–365).
 * Formula: d = 1 − 0.01670963 × cos(2π(doy−3)/365)
 * Same convention as i.hyper.smac's radtran.py. */
double sixs_earth_sun_dist2(int doy)
{
    double beta = 2.0 * M_PI * (doy - 3) / 365.0;
    double d    = 1.0 - 0.01670963 * cos(beta);
    return d * d;
}
