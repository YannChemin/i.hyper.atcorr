/* Standard atmosphere profiles — ported from 6SV2.1 US62.f, MIDSUM.f etc. */
#include "../include/sixs_ctx.h"
#include <string.h>

/* ─── US Standard 1962 ────────────────────────────────────────────────────── */
static const float z6[NATM] = {
    0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,
    18.,19.,20.,21.,22.,23.,24.,25.,30.,35.,40.,45.,50.,70.,100.,99999.
};
static const float p6[NATM] = {
    1.013e3f,8.986e2f,7.950e2f,7.012e2f,6.166e2f,5.405e2f,4.722e2f,4.111e2f,
    3.565e2f,3.080e2f,2.650e2f,2.270e2f,1.940e2f,1.658e2f,1.417e2f,1.211e2f,
    1.035e2f,8.850e1f,7.565e1f,6.467e1f,5.529e1f,4.729e1f,4.047e1f,3.467e1f,
    2.972e1f,2.549e1f,1.197e1f,5.746e0f,2.871e0f,1.491e0f,7.978e-1f,
    5.520e-2f,3.008e-4f,0.0f
};
static const float t6[NATM] = {
    2.881e2f,2.816e2f,2.751e2f,2.687e2f,2.622e2f,2.557e2f,2.492e2f,2.427e2f,
    2.362e2f,2.297e2f,2.232e2f,2.168e2f,2.166e2f,2.166e2f,2.166e2f,2.166e2f,
    2.166e2f,2.166e2f,2.166e2f,2.166e2f,2.166e2f,2.176e2f,2.186e2f,2.196e2f,
    2.206e2f,2.216e2f,2.265e2f,2.365e2f,2.534e2f,2.642e2f,2.706e2f,
    2.197e2f,2.100e2f,2.100e2f
};
static const float wh6[NATM] = {
    5.900e0f,4.200e0f,2.900e0f,1.800e0f,1.100e0f,6.400e-1f,3.800e-1f,2.100e-1f,
    1.200e-1f,4.600e-2f,1.800e-2f,8.200e-3f,3.700e-3f,1.800e-3f,8.400e-4f,
    7.200e-4f,6.100e-4f,5.200e-4f,4.400e-4f,4.400e-4f,4.400e-4f,4.800e-4f,
    5.200e-4f,5.700e-4f,6.100e-4f,6.600e-4f,3.800e-4f,1.600e-4f,6.700e-5f,
    3.200e-5f,1.200e-5f,1.500e-7f,1.000e-9f,0.0f
};
static const float wo6[NATM] = {
    5.400e-5f,5.400e-5f,5.400e-5f,5.000e-5f,4.600e-5f,4.600e-5f,4.500e-5f,
    4.900e-5f,5.200e-5f,7.100e-5f,9.000e-5f,1.300e-4f,1.600e-4f,1.700e-4f,
    1.900e-4f,2.100e-4f,2.400e-4f,2.800e-4f,3.200e-4f,3.500e-4f,3.800e-4f,
    3.800e-4f,3.900e-4f,3.800e-4f,3.600e-4f,3.400e-4f,2.000e-4f,1.100e-4f,
    4.900e-5f,1.700e-5f,4.000e-6f,8.600e-8f,4.300e-11f,0.0f
};

/* ─── Mid-latitude summer ─────────────────────────────────────────────────── */
static const float zms[NATM] = {
    0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,
    18.,19.,20.,21.,22.,23.,24.,25.,30.,35.,40.,45.,50.,70.,100.,99999.
};
static const float pms[NATM] = {
    1.013e3f,9.020e2f,8.020e2f,7.100e2f,6.280e2f,5.540e2f,4.870e2f,4.260e2f,
    3.720e2f,3.240e2f,2.810e2f,2.430e2f,2.090e2f,1.790e2f,1.530e2f,1.310e2f,
    1.110e2f,9.500e1f,8.120e1f,6.950e1f,5.950e1f,5.100e1f,4.370e1f,3.760e1f,
    3.220e1f,2.770e1f,1.320e1f,6.520e0f,3.330e0f,1.760e0f,9.510e-1f,
    6.710e-2f,3.000e-4f,0.0f
};
static const float tms[NATM] = {
    2.940e2f,2.900e2f,2.850e2f,2.790e2f,2.730e2f,2.670e2f,2.610e2f,2.545e2f,
    2.480e2f,2.420e2f,2.355e2f,2.290e2f,2.220e2f,2.160e2f,2.160e2f,2.160e2f,
    2.160e2f,2.160e2f,2.160e2f,2.175e2f,2.190e2f,2.210e2f,2.250e2f,2.290e2f,
    2.330e2f,2.370e2f,2.590e2f,2.730e2f,2.770e2f,2.690e2f,2.550e2f,
    2.190e2f,2.100e2f,2.100e2f
};
static const float whms[NATM] = {
    1.400e1f,9.300e0f,5.900e0f,3.300e0f,1.900e0f,1.000e0f,6.100e-1f,3.700e-1f,
    2.100e-1f,1.200e-1f,6.400e-2f,2.200e-2f,6.000e-3f,1.800e-3f,1.000e-3f,
    7.600e-4f,6.400e-4f,5.600e-4f,5.000e-4f,4.900e-4f,4.500e-4f,5.100e-4f,
    5.100e-4f,5.400e-4f,6.000e-4f,6.700e-4f,3.600e-4f,1.100e-4f,4.300e-5f,
    1.900e-5f,6.300e-6f,1.400e-7f,1.000e-9f,0.0f
};
static const float woms[NATM] = {
    6.000e-5f,6.000e-5f,6.000e-5f,6.200e-5f,6.400e-5f,6.600e-5f,6.900e-5f,
    7.500e-5f,7.900e-5f,1.100e-4f,1.500e-4f,2.100e-4f,2.700e-4f,2.900e-4f,
    3.200e-4f,3.400e-4f,3.800e-4f,4.100e-4f,4.300e-4f,4.500e-4f,4.300e-4f,
    4.300e-4f,3.900e-4f,3.600e-4f,3.400e-4f,3.200e-4f,1.500e-4f,9.200e-5f,
    4.100e-5f,1.300e-5f,4.300e-6f,8.600e-8f,4.300e-11f,0.0f
};

/* Initialize atmosphere in context from model index */
void sixs_init_atmosphere(SixsCtx *ctx, int atmo_model) {
    const float *z, *p, *t, *wh, *wo;
    switch (atmo_model) {
        case 2:  z=zms; p=pms; t=tms; wh=whms; wo=woms; break;
        default: z=z6;  p=p6;  t=t6;  wh=wh6;  wo=wo6; break;
    }
    for (int i = 0; i < NATM; i++) {
        ctx->atm.z[i]  = z[i];
        ctx->atm.p[i]  = p[i];
        ctx->atm.t[i]  = t[i];
        ctx->atm.wh[i] = wh[i];
        ctx->atm.wo[i] = wo[i];
    }
    /* Rayleigh depolarization factor for air (6SV standard) */
    ctx->del.delta = 0.0279f;
    ctx->del.sigma = 0.0f;
    /* Default quadrature */
    ctx->quad.nquad = NQ_P;
    ctx->multi.igmax = 20;
    ctx->err.ier = false;
}
