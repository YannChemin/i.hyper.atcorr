/****************************************************************************
 * MODULE:      i.hyper.atcorr
 * AUTHOR:      i.hyper.smac project
 * PURPOSE:     6SV2.1-based atmospheric correction for hyperspectral imagery.
 *
 * Two operational modes (at least one output must be requested):
 *
 *   lut=   — write a binary LUT file [R_atm, T_down, T_up, s_alb] on an
 *             [AOD × H2O × wavelength] grid for use by i.hyper.smac.
 *
 *   input= / output= — correct a Raster3D radiance cube to surface (BOA)
 *             reflectance using the computed LUT.  Band centre wavelengths are
 *             read from the map's r3.info metadata; fallback to the LUT grid.
 *
 * Physics (6SV standard):
 *   ρ_toa = (π × L × d²) / (E₀ × cos θs)
 *   ρ_boa = (ρ_toa − R_atm) / (T_down × T_up × (1 + s_alb × ρ_boa))
 *
 * Units: L in W/(m² sr µm), E₀ from 6SV2.1 Thuillier solar spectrum
 *        (W/(m² µm)).  No libRadtran dependency required.
 *
 * LUT file format (host-endian binary):
 *   magic     uint32  0x4C555400 ("LUT\0")
 *   version   uint32  1
 *   n_aod, n_h2o, n_wl  int32 each
 *   aod[n_aod]  float32
 *   h2o[n_h2o]  float32
 *   wl [n_wl]   float32  (µm)
 *   R_atm, T_down, T_up, s_alb  float32[n_aod*n_h2o*n_wl] each (C order)
 ****************************************************************************/

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster3d.h>

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/atcorr.h"
#include "include/solar_table.h"

#define LUT_MAGIC   0x4C555400u
#define LUT_VERSION 1u

/* ── Helpers ──────────────────────────────────────────────────────────────── */

static int parse_csv_floats(const char *str, float *out, int max_n)
{
    char *buf = G_store(str);
    char *tok, *saveptr = NULL;
    int   n = 0;

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

static void write_u32(FILE *fp, unsigned int v) { fwrite(&v, 4, 1, fp); }
static void write_i32(FILE *fp, int v)          { fwrite(&v, 4, 1, fp); }
static void write_f32a(FILE *fp, const float *v, size_t n)
{
    fwrite(v, sizeof(float), n, fp);
}

/* ── Band wavelength parsing from r3.info -h ─────────────────────────────── */

/* Parse per-band centre wavelengths from `r3.info -h` output.
 * Looks for lines:  "  Band N: WL nm, FWHM: F nm"
 * Fills wl[0..max_n-1] in µm (-1 = not found).  Returns count parsed. */
static int parse_band_wl(const char *mapname, float *wl, int max_n)
{
    const char *gisbase = getenv("GISBASE");
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s/bin/r3.info -h map=%s 2>/dev/null",
             gisbase ? gisbase : "", mapname);

    FILE *fp = popen(cmd, "r");
    if (!fp) return 0;

    for (int i = 0; i < max_n; i++) wl[i] = -1.0f;

    char line[512];
    int  n = 0;
    while (fgets(line, sizeof(line), fp)) {
        int    bnum;
        double wl_nm;
        if (sscanf(line, " Band %d: %lf nm", &bnum, &wl_nm) == 2) {
            if (bnum >= 1 && bnum <= max_n) {
                wl[bnum - 1] = (float)(wl_nm * 1e-3); /* nm → µm */
                n++;
            }
        }
    }
    pclose(fp);
    return n;
}

/* ── 1-D linear interpolation into a LUT spectral slice ─────────────────── */

/* Interpolate arr[0..n-1] (sampled at lut_wl[]) to a target wavelength.
 * Clamps to boundary values for out-of-range wl_um. */
static float interp_wl(const float *arr, const float *lut_wl, int n, float wl_um)
{
    if (n <= 0)                  return 0.0f;
    if (wl_um <= lut_wl[0])     return arr[0];
    if (wl_um >= lut_wl[n - 1]) return arr[n - 1];
    int i = 0;
    while (i < n - 2 && lut_wl[i + 1] <= wl_um) i++;
    float t = (wl_um - lut_wl[i]) / (lut_wl[i + 1] - lut_wl[i]);
    return arr[i] * (1.0f - t) + arr[i + 1] * t;
}

/* ── Atmospheric correction of a Raster3D cube ───────────────────────────── */

static void correct_raster3d(const char *input_name, const char *output_name,
                              const LutConfig *cfg, const LutArrays *lut,
                              float aod_val, float h2o_val,
                              int doy, float sza_deg)
{
    /* ── Open input map ── */
    const char *in_mapset = G_find_raster3d(input_name, "");
    if (!in_mapset)
        G_fatal_error(_("3D raster map <%s> not found"), input_name);

    RASTER3D_Region region;
    Rast3d_get_window(&region);

    RASTER3D_Map *inmap = Rast3d_open_cell_old(
        input_name, in_mapset, &region,
        RASTER3D_TILE_SAME_AS_FILE, RASTER3D_USE_CACHE_DEFAULT);
    if (!inmap)
        G_fatal_error(_("Cannot open 3D raster <%s>"), input_name);

    /* Use the map's own region for rows/cols/depths */
    Rast3d_get_region_struct_map(inmap, &region);
    int nrows   = region.rows;
    int ncols   = region.cols;
    int ndepths = region.depths;

    G_verbose_message(_("Input <%s>: %d rows × %d cols × %d bands"),
                      input_name, nrows, ncols, ndepths);

    /* ── Parse per-band wavelengths from r3.info metadata ── */
    float *band_wl = G_malloc(ndepths * sizeof(float));
    int    n_parsed = parse_band_wl(input_name, band_wl, ndepths);

    if (n_parsed == 0) {
        G_warning(_("No wavelength metadata in <%s>; "
                    "using LUT grid wavelengths (may not match bands)"),
                  input_name);
        for (int b = 0; b < ndepths; b++)
            band_wl[b] = (b < cfg->n_wl) ? cfg->wl[b]
                                          : cfg->wl[cfg->n_wl - 1];
    } else if (n_parsed < ndepths) {
        G_warning(_("Parsed %d of %d band wavelengths from <%s>"),
                  n_parsed, ndepths, input_name);
    } else {
        G_verbose_message(_("Band wavelengths: %.4f–%.4f µm"),
                          band_wl[0], band_wl[ndepths - 1]);
    }

    /* Warn if any band falls outside the LUT spectral range */
    float wl_lut_min = cfg->wl[0], wl_lut_max = cfg->wl[cfg->n_wl - 1];
    int n_out = 0;
    for (int b = 0; b < ndepths; b++)
        if (band_wl[b] < wl_lut_min || band_wl[b] > wl_lut_max) n_out++;
    if (n_out > 0)
        G_warning(_("%d bands outside LUT spectral range [%.4f, %.4f] µm; "
                    "boundary LUT values will be used"),
                  n_out, wl_lut_min, wl_lut_max);

    /* ── Pre-compute LUT correction slice at (aod_val, h2o_val) ── */
    int    n_wl = cfg->n_wl;
    float *Rs   = G_malloc(n_wl * sizeof(float));
    float *Tds  = G_malloc(n_wl * sizeof(float));
    float *Tus  = G_malloc(n_wl * sizeof(float));
    float *ss   = G_malloc(n_wl * sizeof(float));
    atcorr_lut_slice(cfg, lut, aod_val, h2o_val, Rs, Tds, Tus, ss);

    G_verbose_message(
        _("Correction slice at AOD=%.3f, H2O=%.2f g/cm²: "
          "R_atm(550nm)=%.4f  T_down=%.4f  T_up=%.4f  s=%.4f"),
        aod_val, h2o_val,
        interp_wl(Rs,  cfg->wl, n_wl, 0.55f),
        interp_wl(Tds, cfg->wl, n_wl, 0.55f),
        interp_wl(Tus, cfg->wl, n_wl, 0.55f),
        interp_wl(ss,  cfg->wl, n_wl, 0.55f));

    /* ── Physical constants ── */
    double d2      = sixs_earth_sun_dist2(doy);
    double cos_sza = cos(sza_deg * (M_PI / 180.0));

    /* ── Open output map (same region as input) ── */
    RASTER3D_Map *outmap = Rast3d_open_new_opt_tile_size(
        output_name, RASTER3D_USE_CACHE_X, &region, DCELL_TYPE, 32);
    if (!outmap)
        G_fatal_error(_("Cannot create 3D raster <%s>"), output_name);

    Rast3d_min_unlocked(outmap, RASTER3D_USE_CACHE_X);

    /* ── Correction loop: depth (band) × row × col ── */
    G_message(_("Correcting %d bands × %d rows × %d cols "
                "(AOD=%.3f, H2O=%.2f g/cm², DOY=%d)..."),
              ndepths, nrows, ncols, aod_val, h2o_val, doy);

    double null_d;
    Rast_set_d_null_value(&null_d, 1);

    for (int z = 0; z < ndepths; z++) {  /* z=0: bottom = first band */
        G_percent(z, ndepths, 2);

        float  wl  = band_wl[z];           /* µm */
        float  E0  = sixs_E0(wl);          /* W/(m² µm) */
        float  R_a = interp_wl(Rs,  cfg->wl, n_wl, wl);
        float  T_d = interp_wl(Tds, cfg->wl, n_wl, wl);
        float  T_u = interp_wl(Tus, cfg->wl, n_wl, wl);
        float  s_a = interp_wl(ss,  cfg->wl, n_wl, wl);

        for (int row = 0; row < nrows; row++) {
            for (int col = 0; col < ncols; col++) {
                double L = Rast3d_get_double(inmap, col, row, z);

                if (Rast_is_d_null_value(&L) || E0 <= 0.0) {
                    Rast3d_put_double(outmap, col, row, z, null_d);
                    continue;
                }

                /* TOA reflectance (L in W/(m² sr µm), E0 in W/(m² µm)) */
                double rho_toa = (M_PI * L * d2) / (E0 * cos_sza);

                /* Surface reflectance via 6SV inversion */
                double rho_boa = (double)atcorr_invert(
                    (float)rho_toa, R_a, T_d, T_u, s_a);

                Rast3d_put_double(outmap, col, row, z, rho_boa);
            }
        }
    }
    G_percent(1, 1, 1);

    /* ── Close maps ── */
    if (!Rast3d_close(inmap))
        G_fatal_error(_("Cannot close input map <%s>"), input_name);
    if (!Rast3d_close(outmap))
        G_fatal_error(_("Cannot close output map <%s>"), output_name);

    G_free(band_wl);
    G_free(Rs); G_free(Tds); G_free(Tus); G_free(ss);

    G_message(_("Surface reflectance written to <%s>"), output_name);
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    G_gisinit(argv[0]);
    Rast3d_init_defaults();

    struct GModule *module = G_define_module();
    G_add_keyword(_("imagery"));
    G_add_keyword(_("atmospheric correction"));
    G_add_keyword(_("radiative transfer"));
    G_add_keyword(_("6SV"));
    G_add_keyword(_("LUT"));
    G_add_keyword(_("hyperspectral"));
    module->description =
        _("Atmospheric correction of hyperspectral Raster3D radiance imagery "
          "using the 6SV2.1 radiative transfer algorithm.");

    /* ── I/O ── */
    struct Option *opt_input = G_define_standard_option(G_OPT_R3_INPUT);
    opt_input->required    = NO;
    opt_input->label       = _("Input radiance Raster3D map");
    opt_input->description = _("TOA radiance in W/(m² sr µm); "
                               "band wavelengths read from r3.info metadata");
    opt_input->guisection  = _("Correction");

    struct Option *opt_output = G_define_standard_option(G_OPT_R3_OUTPUT);
    opt_output->required    = NO;
    opt_output->label       = _("Output surface reflectance Raster3D map");
    opt_output->guisection  = _("Correction");

    struct Option *opt_lut = G_define_option();
    opt_lut->key         = "lut";
    opt_lut->type        = TYPE_STRING;
    opt_lut->required    = NO;
    opt_lut->gisprompt   = "new,file,file";
    opt_lut->label       = _("Output LUT binary file");
    opt_lut->description = _("Binary LUT (*.lut) with R_atm, T_down, T_up, s_alb "
                             "on an [AOD × H2O × wavelength] grid");
    opt_lut->guisection  = _("LUT");

    /* ── Geometry ── */
    struct Option *opt_sza = G_define_option();
    opt_sza->key         = "sza";
    opt_sza->type        = TYPE_DOUBLE;
    opt_sza->required    = YES;
    opt_sza->label       = _("Solar zenith angle (degrees, 0–89)");
    opt_sza->guisection  = _("Geometry");

    struct Option *opt_vza = G_define_option();
    opt_vza->key         = "vza";
    opt_vza->type        = TYPE_DOUBLE;
    opt_vza->required    = NO;
    opt_vza->answer      = "0";
    opt_vza->label       = _("View zenith angle (degrees, 0–60)");
    opt_vza->guisection  = _("Geometry");

    struct Option *opt_raa = G_define_option();
    opt_raa->key         = "raa";
    opt_raa->type        = TYPE_DOUBLE;
    opt_raa->required    = NO;
    opt_raa->answer      = "0";
    opt_raa->label       = _("Relative azimuth angle (degrees)");
    opt_raa->guisection  = _("Geometry");

    struct Option *opt_altitude = G_define_option();
    opt_altitude->key         = "altitude";
    opt_altitude->type        = TYPE_DOUBLE;
    opt_altitude->required    = NO;
    opt_altitude->answer      = "1000";
    opt_altitude->label       = _("Sensor altitude (km; > 900 = satellite)");
    opt_altitude->guisection  = _("Geometry");

    /* ── Atmosphere ── */
    struct Option *opt_atmo = G_define_option();
    opt_atmo->key         = "atmosphere";
    opt_atmo->type        = TYPE_STRING;
    opt_atmo->required    = NO;
    opt_atmo->answer      = "us62";
    opt_atmo->options     = "us62,midsum,midwin,tropical,subsum,subwin";
    opt_atmo->description = _("Standard atmosphere model");
    opt_atmo->guisection  = _("Atmosphere");

    struct Option *opt_aerosol = G_define_option();
    opt_aerosol->key         = "aerosol";
    opt_aerosol->type        = TYPE_STRING;
    opt_aerosol->required    = NO;
    opt_aerosol->answer      = "continental";
    opt_aerosol->options     = "none,continental,maritime,urban,desert";
    opt_aerosol->description = _("Aerosol model");
    opt_aerosol->guisection  = _("Atmosphere");

    struct Option *opt_ozone = G_define_option();
    opt_ozone->key         = "ozone";
    opt_ozone->type        = TYPE_DOUBLE;
    opt_ozone->required    = NO;
    opt_ozone->answer      = "300";
    opt_ozone->description = _("Total ozone column (Dobson units)");
    opt_ozone->guisection  = _("Atmosphere");

    /* ── LUT grid ── */
    struct Option *opt_aod = G_define_option();
    opt_aod->key         = "aod";
    opt_aod->type        = TYPE_STRING;
    opt_aod->required    = NO;
    opt_aod->answer      = "0.0,0.05,0.1,0.2,0.4,0.8";
    opt_aod->label       = _("AOD at 550 nm grid (comma-separated)");
    opt_aod->description = _("Aerosol optical depth values for LUT grid points");
    opt_aod->guisection  = _("LUT");

    struct Option *opt_h2o = G_define_option();
    opt_h2o->key         = "h2o";
    opt_h2o->type        = TYPE_STRING;
    opt_h2o->required    = NO;
    opt_h2o->answer      = "0.5,1.0,2.0,3.5,5.0";
    opt_h2o->label       = _("Column water vapour grid in g/cm² (comma-separated)");
    opt_h2o->guisection  = _("LUT");

    struct Option *opt_wl_min = G_define_option();
    opt_wl_min->key         = "wl_min";
    opt_wl_min->type        = TYPE_DOUBLE;
    opt_wl_min->required    = NO;
    opt_wl_min->answer      = "0.40";
    opt_wl_min->description = _("Minimum wavelength (µm)");
    opt_wl_min->guisection  = _("LUT");

    struct Option *opt_wl_max = G_define_option();
    opt_wl_max->key         = "wl_max";
    opt_wl_max->type        = TYPE_DOUBLE;
    opt_wl_max->required    = NO;
    opt_wl_max->answer      = "2.50";
    opt_wl_max->description = _("Maximum wavelength (µm)");
    opt_wl_max->guisection  = _("LUT");

    struct Option *opt_wl_step = G_define_option();
    opt_wl_step->key         = "wl_step";
    opt_wl_step->type        = TYPE_DOUBLE;
    opt_wl_step->required    = NO;
    opt_wl_step->answer      = "0.01";
    opt_wl_step->description = _("Wavelength step (µm)");
    opt_wl_step->guisection  = _("LUT");

    /* ── Correction parameters ── */
    struct Option *opt_doy = G_define_option();
    opt_doy->key         = "doy";
    opt_doy->type        = TYPE_INTEGER;
    opt_doy->required    = NO;
    opt_doy->answer      = "180";
    opt_doy->label       = _("Day of year for Earth-Sun distance (1–365)");
    opt_doy->description = _("Used to compute d²; set to the acquisition date");
    opt_doy->guisection  = _("Correction");

    struct Option *opt_aod_val = G_define_option();
    opt_aod_val->key         = "aod_val";
    opt_aod_val->type        = TYPE_DOUBLE;
    opt_aod_val->required    = NO;
    opt_aod_val->answer      = "0.1";
    opt_aod_val->label       = _("Scene AOD at 550 nm for correction");
    opt_aod_val->description = _("Interpolated from the LUT; should be within the aod= grid");
    opt_aod_val->guisection  = _("Correction");

    struct Option *opt_h2o_val = G_define_option();
    opt_h2o_val->key         = "h2o_val";
    opt_h2o_val->type        = TYPE_DOUBLE;
    opt_h2o_val->required    = NO;
    opt_h2o_val->answer      = "2.0";
    opt_h2o_val->label       = _("Scene column water vapour (g/cm²) for correction");
    opt_h2o_val->description = _("Interpolated from the LUT; should be within the h2o= grid");
    opt_h2o_val->guisection  = _("Correction");

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    /* ── Validate mode ── */
    if (!opt_lut->answer && !opt_output->answer)
        G_fatal_error(_("Specify at least one output: "
                        "lut= (LUT file) or output= (corrected Raster3D)"));
    if (opt_output->answer && !opt_input->answer)
        G_fatal_error(_("output= requires input= (Raster3D radiance map)"));

    /* ── Parse scalar parameters ── */
    float sza      = (float)atof(opt_sza->answer);
    float vza      = (float)atof(opt_vza->answer);
    float raa      = (float)atof(opt_raa->answer);
    float altitude = (float)atof(opt_altitude->answer);
    float ozone    = (float)atof(opt_ozone->answer);
    float wl_min   = (float)atof(opt_wl_min->answer);
    float wl_max   = (float)atof(opt_wl_max->answer);
    float wl_step  = (float)atof(opt_wl_step->answer);
    float aod_val  = (float)atof(opt_aod_val->answer);
    float h2o_val  = (float)atof(opt_h2o_val->answer);
    int   doy      = atoi(opt_doy->answer);

    if (sza < 0.0f || sza >= 90.0f)
        G_fatal_error(_("sza must be in [0, 90)"));
    if (vza < 0.0f || vza > 60.0f)
        G_fatal_error(_("vza must be in [0, 60]"));
    if (wl_step <= 0.0f || wl_min >= wl_max)
        G_fatal_error(_("Invalid wavelength range or step"));
    if (doy < 1 || doy > 365)
        G_fatal_error(_("doy must be in [1, 365]"));

    int atmo_model;
    if      (strcmp(opt_atmo->answer, "us62")     == 0) atmo_model = ATMO_US62;
    else if (strcmp(opt_atmo->answer, "midsum")   == 0) atmo_model = ATMO_MIDSUM;
    else if (strcmp(opt_atmo->answer, "midwin")   == 0) atmo_model = ATMO_MIDWIN;
    else if (strcmp(opt_atmo->answer, "tropical") == 0) atmo_model = ATMO_TROPICAL;
    else if (strcmp(opt_atmo->answer, "subsum")   == 0) atmo_model = ATMO_SUBSUM;
    else if (strcmp(opt_atmo->answer, "subwin")   == 0) atmo_model = ATMO_SUBWIN;
    else    G_fatal_error(_("Unknown atmosphere model: %s"), opt_atmo->answer);

    int aerosol_model;
    if      (strcmp(opt_aerosol->answer, "none")        == 0) aerosol_model = AEROSOL_NONE;
    else if (strcmp(opt_aerosol->answer, "continental") == 0) aerosol_model = AEROSOL_CONTINENTAL;
    else if (strcmp(opt_aerosol->answer, "maritime")    == 0) aerosol_model = AEROSOL_MARITIME;
    else if (strcmp(opt_aerosol->answer, "urban")       == 0) aerosol_model = AEROSOL_URBAN;
    else if (strcmp(opt_aerosol->answer, "desert")      == 0) aerosol_model = AEROSOL_DESERT;
    else    G_fatal_error(_("Unknown aerosol model: %s"), opt_aerosol->answer);

    float aod_buf[64]; int n_aod = parse_csv_floats(opt_aod->answer, aod_buf, 64);
    if (n_aod <= 0) G_fatal_error(_("Cannot parse aod= values: %s"), opt_aod->answer);

    float h2o_buf[64]; int n_h2o = parse_csv_floats(opt_h2o->answer, h2o_buf, 64);
    if (n_h2o <= 0) G_fatal_error(_("Cannot parse h2o= values: %s"), opt_h2o->answer);

    int n_wl = (int)((wl_max - wl_min) / wl_step) + 1;
    if (n_wl < 1 || n_wl > 10000)
        G_fatal_error(_("Wavelength grid size %d out of valid range"), n_wl);

    float *wl_buf = G_malloc(n_wl * sizeof(float));
    for (int i = 0; i < n_wl; i++) wl_buf[i] = wl_min + i * wl_step;

    /* ── Build LUT config and allocate arrays ── */
    LutConfig cfg = {
        .wl = wl_buf, .n_wl = n_wl,
        .aod = aod_buf, .n_aod = n_aod,
        .h2o = h2o_buf, .n_h2o = n_h2o,
        .sza = sza, .vza = vza, .raa = raa,
        .altitude_km = altitude,
        .atmo_model = atmo_model, .aerosol_model = aerosol_model,
        .surface_pressure = 0.0f, .ozone_du = ozone,
    };

    size_t n      = (size_t)n_aod * n_h2o * n_wl;
    float *R_atm  = G_malloc(n * sizeof(float));
    float *T_down = G_malloc(n * sizeof(float));
    float *T_up   = G_malloc(n * sizeof(float));
    float *s_alb  = G_malloc(n * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        R_atm[i] = 0.f; T_down[i] = 1.f; T_up[i] = 1.f; s_alb[i] = 0.f;
    }
    LutArrays lut = { R_atm, T_down, T_up, s_alb };

    /* ── Compute LUT ── */
    G_verbose_message(_("Computing LUT: %d AOD × %d H2O × %d wavelengths "
                        "(SZA=%.1f° VZA=%.1f° RAA=%.1f°)..."),
                      n_aod, n_h2o, n_wl, sza, vza, raa);

    int ret = atcorr_compute_lut(&cfg, &lut);
    if (ret != 0)
        G_fatal_error(_("atcorr_compute_lut failed (code %d)"), ret);

    /* Print 550 nm summary */
    if (G_verbose() >= G_verbose_std()) {
        int i550 = 0;
        float best = 1e9f;
        for (int i = 0; i < n_wl; i++) {
            float d = fabsf(wl_buf[i] - 0.55f);
            if (d < best) { best = d; i550 = i; }
        }
        int    ia  = (n_aod > 2 ? 2 : 0), ih = n_h2o / 2;
        size_t idx = ((size_t)ia * n_h2o + ih) * n_wl + i550;
        G_message(_("At %.3f µm, AOD=%.2f, H2O=%.1f g/cm²:  "
                    "R_atm=%.4f  T_down=%.4f  T_up=%.4f  s_alb=%.4f"),
                  wl_buf[i550], aod_buf[ia], h2o_buf[ih],
                  R_atm[idx], T_down[idx], T_up[idx], s_alb[idx]);
    }

    /* ── Write LUT file ── */
    if (opt_lut->answer) {
        FILE *fp = fopen(opt_lut->answer, "wb");
        if (!fp)
            G_fatal_error(_("Cannot open '%s': %s"),
                          opt_lut->answer, strerror(errno));
        write_u32(fp, LUT_MAGIC);  write_u32(fp, LUT_VERSION);
        write_i32(fp, n_aod);      write_i32(fp, n_h2o);  write_i32(fp, n_wl);
        write_f32a(fp, aod_buf, n_aod);
        write_f32a(fp, h2o_buf, n_h2o);
        write_f32a(fp, wl_buf,  n_wl);
        write_f32a(fp, R_atm,  n);
        write_f32a(fp, T_down, n);
        write_f32a(fp, T_up,   n);
        write_f32a(fp, s_alb,  n);
        if (fclose(fp) != 0)
            G_fatal_error(_("Error closing '%s'"), opt_lut->answer);
        G_message(_("LUT written to '%s' (%zu KB)"),
                  opt_lut->answer, (n * 4 * 4 + 256) / 1024);
    }

    /* ── Atmospheric correction ── */
    if (opt_input->answer) {
        correct_raster3d(opt_input->answer, opt_output->answer,
                         &cfg, &lut, aod_val, h2o_val, doy, sza);
    }

    G_free(wl_buf);
    G_free(R_atm); G_free(T_down); G_free(T_up); G_free(s_alb);

    exit(EXIT_SUCCESS);
}
