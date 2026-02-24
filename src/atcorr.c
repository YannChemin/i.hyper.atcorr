/* i.hyper.atcorr — main shared library entry point.
 * Implements atcorr_version() and validates atcorr_compute_lut() inputs.
 * The actual LUT computation is in lut.c. */
#include "../include/atcorr.h"
#include "../include/sixs_ctx.h"
#include <string.h>

const char *atcorr_version(void)
{
    return "i.hyper.atcorr 0.1.0 (6SV2.1 port, " __DATE__ ")";
}
