/* LUT bilinear interpolation in (aod, h2o) space.
 * Produces per-wavelength correction arrays at a fixed (aod, h2o) point
 * for use in the atmospheric correction inversion loop. */
#include "atcorr.h"

/* Fill output arrays [n_wl] with bilinear interpolation of the LUT at
 * (aod_val, h2o_val).  Values outside the grid are clamped to the boundary.
 *
 * Outputs Rs, Tds, Tus, ss must be pre-allocated with cfg->n_wl floats. */
void atcorr_lut_slice(const LutConfig *cfg, const LutArrays *lut,
                      float aod_val, float h2o_val,
                      float *Rs, float *Tds, float *Tus, float *ss)
{
    int na = cfg->n_aod, nh = cfg->n_h2o, nw = cfg->n_wl;

    /* AOD bracket */
    int ia0 = 0;
    while (ia0 < na - 2 && cfg->aod[ia0 + 1] <= aod_val) ia0++;
    int ia1 = (ia0 < na - 1) ? ia0 + 1 : ia0;
    float ta = (ia1 != ia0)
                   ? (aod_val - cfg->aod[ia0]) / (cfg->aod[ia1] - cfg->aod[ia0])
                   : 0.0f;
    if (ta < 0.0f) ta = 0.0f;
    if (ta > 1.0f) ta = 1.0f;

    /* H2O bracket */
    int ih0 = 0;
    while (ih0 < nh - 2 && cfg->h2o[ih0 + 1] <= h2o_val) ih0++;
    int ih1 = (ih0 < nh - 1) ? ih0 + 1 : ih0;
    float th = (ih1 != ih0)
                   ? (h2o_val - cfg->h2o[ih0]) / (cfg->h2o[ih1] - cfg->h2o[ih0])
                   : 0.0f;
    if (th < 0.0f) th = 0.0f;
    if (th > 1.0f) th = 1.0f;

#define IDX(ia, ih, iw) ((size_t)(ia) * nh * nw + (size_t)(ih) * nw + (iw))
#define BI4(arr) \
    ((arr)[IDX(ia0,ih0,iw)] * (1.f-ta)*(1.f-th) + \
     (arr)[IDX(ia1,ih0,iw)] *      ta *(1.f-th) + \
     (arr)[IDX(ia0,ih1,iw)] * (1.f-ta)*      th + \
     (arr)[IDX(ia1,ih1,iw)] *      ta *      th)

    for (int iw = 0; iw < nw; iw++) {
        Rs[iw]  = BI4(lut->R_atm);
        Tds[iw] = BI4(lut->T_down);
        Tus[iw] = BI4(lut->T_up);
        ss[iw]  = BI4(lut->s_alb);
    }
#undef BI4
#undef IDX
}
