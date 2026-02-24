#pragma once
#include "../include/sixs_ctx.h"

/* Interpolate disc quantities from 20 reference wavelengths to wl (µm).
 * ctx->disc must be filled by sixs_discom() first. */
void sixs_interp(const SixsCtx *ctx, int iaer, float wl,
                  float taer55, float taer55p,
                  float *roatm, float *T_down, float *T_up, float *s_alb,
                  float *tray_out, float *taer_out);
