/****************************************************************************
 * MODULE:      i.hyper.atcorr
 * AUTHOR:      i.hyper.smac project
 * PURPOSE:     Compute an atmospheric correction LUT using the 6SV2.1
 *              radiative transfer algorithm (C port with OpenMP).
 *
 * The module writes a binary LUT file containing R_atm, T_down, T_up, and
 * spherical albedo (s) on a grid of [AOD × H2O × wavelength].  The file
 * can be loaded by i.hyper.smac or inspected with --verbose output.
 *
 * LUT file format (little-endian binary):
 *   magic     uint32  0x4C555400 ("LUT\0")
 *   version   uint32  1
 *   n_aod     int32
 *   n_h2o     int32
 *   n_wl      int32
 *   aod[n_aod]        float32
 *   h2o[n_h2o]        float32
 *   wl[n_wl]          float32   (µm)
 *   R_atm  [n_aod*n_h2o*n_wl]  float32
 *   T_down [n_aod*n_h2o*n_wl]  float32
 *   T_up   [n_aod*n_h2o*n_wl]  float32
 *   s_alb  [n_aod*n_h2o*n_wl]  float32
 ****************************************************************************/

#include <grass/gis.h>
#include <grass/glocale.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "include/atcorr.h"

/* LUT file magic number */
#define LUT_MAGIC   0x4C555400u
#define LUT_VERSION 1u

/* ── helpers ─────────────────────────────────────────────────────────────── */

/* Parse a comma-separated list of floats into out[].
 * Returns the count parsed, or -1 on error. */
static int parse_csv_floats(const char *str, float *out, int max_n)
{
    char *buf = G_store(str);
    char *tok, *saveptr = NULL;
    int n = 0;

    tok = strtok_r(buf, ",", &saveptr);
    while (tok && n < max_n) {
        char *end;
        out[n] = (float)strtod(tok, &end);
        if (end == tok) { G_free(buf); return -1; }
        n++;
        tok = strtok_r(NULL, ",", &saveptr);
    }
    G_free(buf);
    return n;
}

/* Write a uint32 in host byte order (the format is host-native). */
static void write_u32(FILE *fp, unsigned int v)
{
    fwrite(&v, sizeof(v), 1, fp);
}
static void write_i32(FILE *fp, int v)
{
    fwrite(&v, sizeof(v), 1, fp);
}
static void write_f32_array(FILE *fp, const float *v, size_t n)
{
    fwrite(v, sizeof(float), n, fp);
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    /* ── GRASS initialisation ─────────────────────────────────────────────── */
    G_gisinit(argv[0]);

    struct GModule *module = G_define_module();
    G_add_keyword(_("imagery"));
    G_add_keyword(_("atmospheric correction"));
    G_add_keyword(_("radiative transfer"));
    G_add_keyword(_("6SV"));
    G_add_keyword(_("LUT"));
    module->description =
        _("Computes an atmospheric correction look-up table (LUT) "
          "using the 6SV2.1 radiative transfer algorithm.");

    /* ── Options ──────────────────────────────────────────────────────────── */
    struct Option *opt_output = G_define_standard_option(G_OPT_F_OUTPUT);
    opt_output->description = _("Output LUT binary file (*.lut)");

    struct Option *opt_sza = G_define_option();
    opt_sza->key         = "sza";
    opt_sza->type        = TYPE_DOUBLE;
    opt_sza->required    = YES;
    opt_sza->label       = _("Solar zenith angle");
    opt_sza->description = _("Solar zenith angle in degrees (0–89)");

    struct Option *opt_vza = G_define_option();
    opt_vza->key         = "vza";
    opt_vza->type        = TYPE_DOUBLE;
    opt_vza->required    = NO;
    opt_vza->answer      = "0";
    opt_vza->label       = _("View zenith angle");
    opt_vza->description = _("Sensor view zenith angle in degrees (0–60)");

    struct Option *opt_raa = G_define_option();
    opt_raa->key         = "raa";
    opt_raa->type        = TYPE_DOUBLE;
    opt_raa->required    = NO;
    opt_raa->answer      = "0";
    opt_raa->label       = _("Relative azimuth angle");
    opt_raa->description = _("Relative azimuth angle between sun and sensor (degrees)");

    struct Option *opt_altitude = G_define_option();
    opt_altitude->key         = "altitude";
    opt_altitude->type        = TYPE_DOUBLE;
    opt_altitude->required    = NO;
    opt_altitude->answer      = "1000";
    opt_altitude->label       = _("Sensor altitude");
    opt_altitude->description = _("Sensor altitude in km (>900 treated as satellite)");

    struct Option *opt_atmo = G_define_option();
    opt_atmo->key         = "atmosphere";
    opt_atmo->type        = TYPE_STRING;
    opt_atmo->required    = NO;
    opt_atmo->answer      = "us62";
    opt_atmo->options     = "us62,midsum,midwin,tropical,subsum,subwin";
    opt_atmo->description = _("Standard atmosphere model");

    struct Option *opt_aerosol = G_define_option();
    opt_aerosol->key         = "aerosol";
    opt_aerosol->type        = TYPE_STRING;
    opt_aerosol->required    = NO;
    opt_aerosol->answer      = "continental";
    opt_aerosol->options     = "none,continental,maritime,urban,desert";
    opt_aerosol->description = _("Aerosol model");

    struct Option *opt_ozone = G_define_option();
    opt_ozone->key         = "ozone";
    opt_ozone->type        = TYPE_DOUBLE;
    opt_ozone->required    = NO;
    opt_ozone->answer      = "300";
    opt_ozone->label       = _("Ozone column");
    opt_ozone->description = _("Total ozone column in Dobson units");

    struct Option *opt_aod = G_define_option();
    opt_aod->key         = "aod";
    opt_aod->type        = TYPE_STRING;
    opt_aod->required    = NO;
    opt_aod->answer      = "0.0,0.05,0.1,0.2,0.4,0.8";
    opt_aod->label       = _("AOD grid");
    opt_aod->description = _("Comma-separated AOD at 550 nm values for the LUT grid");

    struct Option *opt_h2o = G_define_option();
    opt_h2o->key         = "h2o";
    opt_h2o->type        = TYPE_STRING;
    opt_h2o->required    = NO;
    opt_h2o->answer      = "0.5,1.0,2.0,3.5,5.0";
    opt_h2o->label       = _("Water vapour grid");
    opt_h2o->description = _("Comma-separated column water vapour values (g/cm²)");

    struct Option *opt_wl_min = G_define_option();
    opt_wl_min->key         = "wl_min";
    opt_wl_min->type        = TYPE_DOUBLE;
    opt_wl_min->required    = NO;
    opt_wl_min->answer      = "0.40";
    opt_wl_min->label       = _("Minimum wavelength");
    opt_wl_min->description = _("Minimum wavelength in micrometres");

    struct Option *opt_wl_max = G_define_option();
    opt_wl_max->key         = "wl_max";
    opt_wl_max->type        = TYPE_DOUBLE;
    opt_wl_max->required    = NO;
    opt_wl_max->answer      = "2.50";
    opt_wl_max->label       = _("Maximum wavelength");
    opt_wl_max->description = _("Maximum wavelength in micrometres");

    struct Option *opt_wl_step = G_define_option();
    opt_wl_step->key         = "wl_step";
    opt_wl_step->type        = TYPE_DOUBLE;
    opt_wl_step->required    = NO;
    opt_wl_step->answer      = "0.01";
    opt_wl_step->label       = _("Wavelength step");
    opt_wl_step->description = _("Wavelength step in micrometres");

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    /* ── Parse parameters ────────────────────────────────────────────────── */
    float sza      = (float)atof(opt_sza->answer);
    float vza      = (float)atof(opt_vza->answer);
    float raa      = (float)atof(opt_raa->answer);
    float altitude = (float)atof(opt_altitude->answer);
    float ozone    = (float)atof(opt_ozone->answer);
    float wl_min   = (float)atof(opt_wl_min->answer);
    float wl_max   = (float)atof(opt_wl_max->answer);
    float wl_step  = (float)atof(opt_wl_step->answer);

    /* Validate geometry */
    if (sza < 0.0f || sza >= 90.0f)
        G_fatal_error(_("Solar zenith angle must be in [0, 90)"));
    if (vza < 0.0f || vza > 60.0f)
        G_fatal_error(_("View zenith angle must be in [0, 60]"));
    if (wl_step <= 0.0f || wl_min >= wl_max)
        G_fatal_error(_("Invalid wavelength range or step"));

    /* Atmosphere model */
    int atmo_model;
    if      (strcmp(opt_atmo->answer, "us62")     == 0) atmo_model = ATMO_US62;
    else if (strcmp(opt_atmo->answer, "midsum")   == 0) atmo_model = ATMO_MIDSUM;
    else if (strcmp(opt_atmo->answer, "midwin")   == 0) atmo_model = ATMO_MIDWIN;
    else if (strcmp(opt_atmo->answer, "tropical") == 0) atmo_model = ATMO_TROPICAL;
    else if (strcmp(opt_atmo->answer, "subsum")   == 0) atmo_model = ATMO_SUBSUM;
    else if (strcmp(opt_atmo->answer, "subwin")   == 0) atmo_model = ATMO_SUBWIN;
    else    G_fatal_error(_("Unknown atmosphere model: %s"), opt_atmo->answer);

    /* Aerosol model */
    int aerosol_model;
    if      (strcmp(opt_aerosol->answer, "none")        == 0) aerosol_model = AEROSOL_NONE;
    else if (strcmp(opt_aerosol->answer, "continental") == 0) aerosol_model = AEROSOL_CONTINENTAL;
    else if (strcmp(opt_aerosol->answer, "maritime")    == 0) aerosol_model = AEROSOL_MARITIME;
    else if (strcmp(opt_aerosol->answer, "urban")       == 0) aerosol_model = AEROSOL_URBAN;
    else if (strcmp(opt_aerosol->answer, "desert")      == 0) aerosol_model = AEROSOL_DESERT;
    else    G_fatal_error(_("Unknown aerosol model: %s"), opt_aerosol->answer);

    /* AOD grid */
    float aod_buf[64];
    int n_aod = parse_csv_floats(opt_aod->answer, aod_buf, 64);
    if (n_aod <= 0)
        G_fatal_error(_("Could not parse AOD values: %s"), opt_aod->answer);

    /* H2O grid */
    float h2o_buf[64];
    int n_h2o = parse_csv_floats(opt_h2o->answer, h2o_buf, 64);
    if (n_h2o <= 0)
        G_fatal_error(_("Could not parse H2O values: %s"), opt_h2o->answer);

    /* Wavelength grid */
    int n_wl = (int)((wl_max - wl_min) / wl_step) + 1;
    if (n_wl < 1 || n_wl > 10000)
        G_fatal_error(_("Wavelength grid has invalid size (%d)"), n_wl);
    float *wl_buf = G_malloc(n_wl * sizeof(float));
    for (int i = 0; i < n_wl; i++)
        wl_buf[i] = wl_min + i * wl_step;

    /* ── Build config ────────────────────────────────────────────────────── */
    LutConfig cfg = {
        .wl            = wl_buf,
        .n_wl          = n_wl,
        .aod           = aod_buf,
        .n_aod         = n_aod,
        .h2o           = h2o_buf,
        .n_h2o         = n_h2o,
        .sza           = sza,
        .vza           = vza,
        .raa           = raa,
        .altitude_km   = altitude,
        .atmo_model    = atmo_model,
        .aerosol_model = aerosol_model,
        .surface_pressure = 0.0f,
        .ozone_du      = ozone,
    };

    size_t n = (size_t)n_aod * n_h2o * n_wl;

    float *R_atm  = G_malloc(n * sizeof(float));
    float *T_down = G_malloc(n * sizeof(float));
    float *T_up   = G_malloc(n * sizeof(float));
    float *s_alb  = G_malloc(n * sizeof(float));

    /* initialise T to 1 (no atmosphere) so partial failures are visible */
    for (size_t i = 0; i < n; i++) { R_atm[i]=0.f; T_down[i]=1.f; T_up[i]=1.f; s_alb[i]=0.f; }

    LutArrays out = { R_atm, T_down, T_up, s_alb };

    /* ── Compute LUT ──────────────────────────────────────────────────────── */
    G_verbose_message(_("Computing LUT: %d AOD × %d H2O × %d wavelengths "
                        "(SZA=%.1f° VZA=%.1f° RAA=%.1f°) …"),
                      n_aod, n_h2o, n_wl, sza, vza, raa);

    int ret = atcorr_compute_lut(&cfg, &out);
    if (ret != 0)
        G_fatal_error(_("atcorr_compute_lut failed (code %d)"), ret);

    /* ── Print summary ──────────────────────────────────────────────────────*/
    if (G_verbose() >= G_verbose_std()) {
        /* Find 550nm index */
        int i550 = 0;
        float best = 1e9f;
        for (int i = 0; i < n_wl; i++) {
            float d = fabsf(wl_buf[i] - 0.55f);
            if (d < best) { best = d; i550 = i; }
        }
        G_message(_("At %.3f µm, AOD=%.2f, H2O=%.1f g/cm²:"),
                  wl_buf[i550],
                  aod_buf[n_aod > 2 ? 2 : 0],
                  h2o_buf[n_h2o / 2]);
        int ia = (n_aod > 2 ? 2 : 0), ih = n_h2o / 2;
        size_t idx = ((size_t)ia * n_h2o + ih) * n_wl + i550;
        G_message(_("  R_atm=%.4f  T_down=%.4f  T_up=%.4f  s_alb=%.4f"),
                  R_atm[idx], T_down[idx], T_up[idx], s_alb[idx]);
    }

    /* ── Write LUT file ───────────────────────────────────────────────────── */
    FILE *fp = fopen(opt_output->answer, "wb");
    if (!fp)
        G_fatal_error(_("Cannot open output file '%s': %s"),
                      opt_output->answer, strerror(errno));

    write_u32(fp, LUT_MAGIC);
    write_u32(fp, LUT_VERSION);
    write_i32(fp, n_aod);
    write_i32(fp, n_h2o);
    write_i32(fp, n_wl);
    write_f32_array(fp, aod_buf, n_aod);
    write_f32_array(fp, h2o_buf, n_h2o);
    write_f32_array(fp, wl_buf,  n_wl);
    write_f32_array(fp, R_atm,  n);
    write_f32_array(fp, T_down, n);
    write_f32_array(fp, T_up,   n);
    write_f32_array(fp, s_alb,  n);

    if (fclose(fp) != 0)
        G_fatal_error(_("Error closing output file '%s'"), opt_output->answer);

    G_message(_("LUT written to '%s' (%zu KB)"),
              opt_output->answer,
              (n * 4 * 4 + 256) / 1024);

    G_free(wl_buf);
    G_free(R_atm);
    G_free(T_down);
    G_free(T_up);
    G_free(s_alb);

    exit(EXIT_SUCCESS);
}
