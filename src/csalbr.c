/* Rayleigh spherical albedo — ported from 6SV2.1 CSALBR.f */
#include <math.h>

static float fintexp1(float xtau) {
    /* E1 integral approximation, accuracy 2e-7 for 0 < xtau < 1 */
    static const float a[6] = {
        -0.57721566f, 0.99999193f, -0.24991055f,
         0.05519968f, -0.00976004f, 0.00107857f
    };
    float xx = a[0];
    float xf = 1.0f;
    for (int i = 1; i <= 5; i++) {
        xf *= xtau;
        xx += a[i] * xf;
    }
    return xx - logf(xtau);
}

static float fintexp3(float xtau) {
    return (expf(-xtau) * (1.0f - xtau) + xtau * xtau * fintexp1(xtau)) / 2.0f;
}

void sixs_csalbr(float xtau, float *xalb) {
    *xalb = (3.0f * xtau - fintexp3(xtau) * (4.0f + 2.0f * xtau) + 2.0f * expf(-xtau))
            / (4.0f + 3.0f * xtau);
}
