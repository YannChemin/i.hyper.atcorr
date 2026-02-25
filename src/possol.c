/* Solar position — ported from 6SV2.1 POSSOL.f */
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Compute day-of-year from calendar date.
 * jday: day of month (1-31); month: 1-12; ia: year (0 = non-leap). */
static int day_number(int jday, int month, int ia)
{
    int j;
    if (month <= 2) {
        j = 31 * (month - 1) + jday;
    } else if (month > 8) {
        j = 31 * (month - 1) - (month - 2) / 2 - 2 + jday;
    } else {
        j = 31 * (month - 1) - (month - 1) / 2 - 2 + jday;
    }
    if (ia != 0 && (ia % 4) == 0) j += 1;
    return j;
}

/* Compute solar zenith (asol) and azimuth (phi0) angles via Fourier series.
 * j: day of year (1-365); tu: UTC decimal hours;
 * xlon: longitude deg (E positive); xlat: latitude deg (N positive). */
static void pos_fft(int j, float tu, float xlon, float xlat,
                    float *asol, float *phi0)
{
    const float fac = (float)(M_PI / 180.0);
    const float pi  = (float)M_PI;

    float tsm = tu + xlon / 15.0f;
    float xla = xlat * fac;
    float xj  = (float)j;
    float tet = 2.0f * pi * xj / 365.0f;

    /* Time equation (minutes decimal) */
    float et = 0.000075f
             + 0.001868f * cosf(tet) - 0.032077f * sinf(tet)
             - 0.014615f * cosf(2.0f * tet) - 0.040849f * sinf(2.0f * tet);
    et = et * 12.0f * 60.0f / pi;

    /* True solar time, hour angle */
    float tsv = tsm + et / 60.0f - 12.0f;
    float ah  = tsv * 15.0f * fac;

    /* Solar declination (radians) */
    float delta = 0.006918f
                - 0.399912f * cosf(tet)  + 0.070257f * sinf(tet)
                - 0.006758f * cosf(2.0f * tet) + 0.000907f * sinf(2.0f * tet)
                - 0.002697f * cosf(3.0f * tet) + 0.001480f * sinf(3.0f * tet);

    /* Elevation and azimuth */
    float amuzero = sinf(xla) * sinf(delta) + cosf(xla) * cosf(delta) * cosf(ah);
    float elev    = asinf(amuzero);
    float az      = cosf(delta) * sinf(ah) / cosf(elev);
    if (fabsf(az) > 1.0f) az = (az > 0.0f) ? 1.0f : -1.0f;
    float caz  = (-cosf(xla) * sinf(delta) + sinf(xla) * cosf(delta) * cosf(ah))
                 / cosf(elev);
    float azim = asinf(az);
    if (caz <= 0.0f) azim = pi - azim;
    if (caz > 0.0f && az <= 0.0f) azim = 2.0f * pi + azim;
    azim += pi;
    if (azim > 2.0f * pi) azim -= 2.0f * pi;

    elev  = elev * 180.0f / pi;
    *asol = 90.0f - elev;
    *phi0 = azim / fac;
}

/* Public API: compute solar zenith angle (asol, degrees) and azimuth (phi0, degrees).
 * month, jday: calendar date; tu: UTC hours; xlon, xlat: degrees (E/N positive).
 * ia: year (for leap-year correction), 0 to ignore. */
void sixs_possol(int month, int jday, float tu, float xlon, float xlat,
                 float *asol, float *phi0, int ia)
{
    int nojour = day_number(jday, month, ia);
    pos_fft(nojour, tu, xlon, xlat, asol, phi0);
}
