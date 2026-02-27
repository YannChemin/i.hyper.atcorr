/**
 * \file atcorr.c
 * \brief Shared library entry point: version string and input validation.
 *
 * Implements atcorr_version() and the input-validation wrapper around
 * atcorr_compute_lut().  The actual LUT computation is in lut.c.
 */
#include "../include/atcorr.h"
#include "../include/sixs_ctx.h"
#include <string.h>

const char *atcorr_version(void)
{
    return "i.hyper.atcorr 0.1.0 (6SV2.1 port, " __DATE__ ")";
}
