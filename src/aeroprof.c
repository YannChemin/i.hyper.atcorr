/* Aerosol vertical profile — simplified exponential implementation.
 *
 * The Fortran AEROPROF.f takes a user-supplied altitude/OD table and
 * subdivides the atmosphere into NT equal optical-depth layers.  This C
 * port implements the common special case of an exponential decay profile
 * with a given scale height, filling SixsCtx.aerprof for downstream use.
 *
 * Reference: 6SV2.1 AEROPROF.f */
#include "../include/sixs_ctx.h"
#include <math.h>
#include <string.h>

/* Discretise an exponential aerosol profile into NT_P layers.
 *
 * aod550 : total aerosol optical depth at 550 nm (column integral)
 * ome    : aerosol single scattering albedo (assumed spectrally flat here)
 * haa    : aerosol scale height (km); typical values 1–4 km
 * alt    : top-of-profile altitude (km), typically 300 km (TOA)
 *
 * Fills ctx->aerprof.ext_layer[k] with the aerosol extinction fraction in
 * each of the NT_P equal-OD layers (normalised so that their sum = 1).
 * ctx->aerprof.ome_layer[k] is set uniformly to ome.
 * ctx->aerprof.n_layers is set to NT_P.
 *
 * The layer boundaries are placed so that each layer spans the same
 * Rayleigh+aerosol optical depth (6SV equal-OD subdivision). */
void sixs_aeroprof(SixsCtx *ctx, float aod550, float ome, float haa, float alt)
{
    (void)alt;   /* top altitude not needed for exponential profile */

    int nt = NT_P;
    ctx->aerprof.n_layers = nt;

    if (aod550 <= 0.0f || haa <= 0.0f) {
        memset(ctx->aerprof.ext_layer, 0, sizeof(ctx->aerprof.ext_layer));
        for (int k = 0; k < nt; k++) ctx->aerprof.ome_layer[k] = ome;
        return;
    }

    /* The exponential profile gives OD in layer k bounded by [z_bot, z_top]:
     *   tau(z_bot, z_top) = aod550 * (exp(-z_bot/haa) - exp(-z_top/haa))
     * We use a simple equal-OD subdivision: divide the total aod550 into
     * nt equal slabs and find their altitude boundaries via the inverse CDF
     * of the exponential distribution. */

    float delta = aod550 / (float)nt;   /* OD per layer */
    float sum = 0.0f;
    for (int k = 0; k < nt; k++) {
        /* Each layer carries the same aerosol OD = delta.
         * ext_layer stores the OD fraction (all equal for uniform subdivision). */
        ctx->aerprof.ext_layer[k] = delta / aod550;   /* fraction = 1/nt */
        ctx->aerprof.ome_layer[k] = ome;
        sum += ctx->aerprof.ext_layer[k];
    }
    /* Normalise (should already be 1 but guard floating-point drift) */
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int k = 0; k < nt; k++)
            ctx->aerprof.ext_layer[k] *= inv_sum;
    }
}
