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
 * ISOFIT-inspired improvements (all optional):
 *   #1 Per-pixel AOD/H2O raster maps + Gaussian spatial smoothing
 *   #2 In-loop Vermote adjacency effect correction (adj_psf= option)
 *   #3 Surface prior MAP regularisation (-r flag)
 *   #4 Per-band reflectance uncertainty output (-u flag, uncertainty= option)
 *   #5 Analytical MAP inner loop via surface_model_regularize()
 *   #6 Per-band model discrepancy floor (added in quadrature to σ² before MAP)
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
#include <grass/raster.h>
#include <grass/raster3d.h>

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/atcorr.h"
#include "include/solar_table.h"
#include "include/spatial.h"
#include "include/adjacency.h"
#include "include/surface_model.h"
#include "include/uncertainty.h"

#define LUT_MAGIC   0x4C555400u
#define LUT_VERSION 1u

/* ── ISOFIT improvement parameters ──────────────────────────────────────────── */

typedef struct {
    /* #1 Per-pixel atmospheric maps + smoothing */
    const char *aod_map;      /* 2-D raster name, or NULL for scalar */
    const char *h2o_map;      /* 2-D raster name, or NULL for scalar */
    float       smooth_sigma; /* Gaussian σ in pixels (0 = disabled) */

    /* #2 Adjacency correction */
    float adj_psf_km;         /* Environmental PSF radius in km (0 = disabled) */
    float pixel_size_m;       /* Pixel size in metres (0 = auto from region) */

    /* #3/#5 Surface prior MAP regularisation */
    int   do_surface_prior;   /* 0 = disabled */

    /* #4/#6 Uncertainty */
    int         do_uncertainty;
    const char *unc_output;   /* Output Raster3D name, or NULL */
} IsoFitParams;

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

static int parse_band_wl(const char *mapname, float *wl, float *fwhm, int max_n)
{
    const char *gisbase = getenv("GISBASE");
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s/bin/r3.info -h map=%s 2>/dev/null",
             gisbase ? gisbase : "", mapname);

    FILE *fp = popen(cmd, "r");
    if (!fp) return 0;

    for (int i = 0; i < max_n; i++) {
        wl[i] = -1.0f;
        if (fwhm) fwhm[i] = 0.0f;
    }

    char line[512];
    int  n = 0;
    while (fgets(line, sizeof(line), fp)) {
        int    bnum;
        double wl_nm;
        if (sscanf(line, " Band %d: %lf nm", &bnum, &wl_nm) == 2) {
            if (bnum >= 1 && bnum <= max_n) {
                wl[bnum - 1] = (float)(wl_nm * 1e-3);
                if (fwhm) {
                    const char *fp_str = strstr(line, "FWHM:");
                    if (fp_str) {
                        double fwhm_nm;
                        if (sscanf(fp_str, "FWHM: %lf nm", &fwhm_nm) == 1)
                            fwhm[bnum - 1] = (float)(fwhm_nm * 1e-3);
                    }
                }
                n++;
            }
        }
    }
    pclose(fp);
    return n;
}

/* ── 1-D linear interpolation ───────────────────────────────────────────────── */

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

/* ── Load a 2-D GRASS raster into a float array ─────────────────────────────── */

static float *load_raster2d(const char *mapname, int nrows, int ncols,
                             float fill_value)
{
    float *data = G_malloc((size_t)nrows * ncols * sizeof(float));

    char *mname = G_store(mapname);   /* non-const copy for G_find_raster */
    const char *mapset = G_find_raster(mname, "");
    G_free(mname);
    if (!mapset) {
        G_warning(_("2-D raster <%s> not found; using fill value %.4f"),
                  mapname, fill_value);
        for (int i = 0; i < nrows * ncols; i++) data[i] = fill_value;
        return data;
    }

    int fd = Rast_open_old(mapname, mapset);
    DCELL *row_buf = Rast_allocate_d_buf();

    for (int r = 0; r < nrows; r++) {
        Rast_get_d_row(fd, row_buf, r);
        for (int c = 0; c < ncols; c++) {
            if (Rast_is_d_null_value(&row_buf[c]))
                data[r * ncols + c] = NAN;
            else
                data[r * ncols + c] = (float)row_buf[c];
        }
    }

    Rast_close(fd);
    G_free(row_buf);

    /* Fill NaN with fill_value */
    int n_nan = 0;
    for (int i = 0; i < nrows * ncols; i++) {
        if (isnan(data[i])) { data[i] = fill_value; n_nan++; }
    }
    if (n_nan > 0)
        G_verbose_message(_("Filled %d NaN pixels in <%s> with %.4f"),
                          n_nan, mapname, fill_value);

    return data;
}

/* ── Write a 2-D band into an open Raster3D map ─────────────────────────────── */

static void write_band_to_rast3d(RASTER3D_Map *outmap,
                                  const float *band, int nrows, int ncols, int z,
                                  double null_d)
{
    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            float v = band[row * ncols + col];
            Rast3d_put_double(outmap, col, row, z,
                              isfinite(v) ? (double)v : null_d);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * correct_raster3d — full correction pipeline with ISOFIT improvements
 * ═══════════════════════════════════════════════════════════════════════════ */

static void correct_raster3d(const char *input_name, const char *output_name,
                              const LutConfig *cfg, LutArrays *lut,
                              float aod_val, float h2o_val,
                              int doy, float sza_deg,
                              const IsoFitParams *iso)
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

    Rast3d_get_region_struct_map(inmap, &region);
    int nrows   = region.rows;
    int ncols   = region.cols;
    int ndepths = region.depths;
    int npix    = nrows * ncols;

    G_verbose_message(_("Input <%s>: %d rows × %d cols × %d bands"),
                      input_name, nrows, ncols, ndepths);

    /* ── Parse per-band wavelengths ── */
    float *band_wl   = G_malloc(ndepths * sizeof(float));
    float *band_fwhm = G_malloc(ndepths * sizeof(float));
    int    n_parsed  = parse_band_wl(input_name, band_wl, band_fwhm, ndepths);

    if (n_parsed == 0) {
        G_warning(_("No wavelength metadata in <%s>; using LUT grid"), input_name);
        for (int b = 0; b < ndepths; b++)
            band_wl[b] = (b < cfg->n_wl) ? cfg->wl[b] : cfg->wl[cfg->n_wl - 1];
    } else if (n_parsed < ndepths) {
        G_warning(_("Parsed %d of %d band wavelengths"), n_parsed, ndepths);
    } else {
        G_verbose_message(_("Band wavelengths: %.4f–%.4f µm"),
                          band_wl[0], band_wl[ndepths - 1]);
    }

    /* ── SRF correction for sub-nm sensors ── */
    {
        float min_fwhm_nm = 1e9f;
        int   n_fwhm = 0;
        for (int b = 0; b < ndepths; b++) {
            if (band_fwhm[b] > 0.0f) {
                n_fwhm++;
                float fn = band_fwhm[b] * 1000.0f;
                if (fn < min_fwhm_nm) min_fwhm_nm = fn;
            }
        }
        if (n_fwhm > 0 && min_fwhm_nm < 5.0f) {
            G_message(_("Sub-nm sensor (min FWHM=%.2f nm): "
                        "applying reptran fine SRF gas correction..."),
                      min_fwhm_nm);
            SrfConfig srf_cfg = { .fwhm_um = band_fwhm, .threshold_um = 0.005f };
            SrfCorrection *srf = atcorr_srf_compute(&srf_cfg, cfg);
            if (srf) {
                atcorr_srf_apply(srf, cfg, lut);
                atcorr_srf_free(srf);
                G_verbose_message(_("SRF gas correction applied."));
            } else {
                G_warning(_("SRF correction failed; proceeding without it"));
            }
        }
    }

    /* Warn on bands outside LUT range */
    {
        int n_out = 0;
        for (int b = 0; b < ndepths; b++)
            if (band_wl[b] < cfg->wl[0] || band_wl[b] > cfg->wl[cfg->n_wl-1])
                n_out++;
        if (n_out > 0)
            G_warning(_("%d bands outside LUT range [%.4f, %.4f] µm"),
                      n_out, cfg->wl[0], cfg->wl[cfg->n_wl-1]);
    }

    /* ── #1: Load and optionally smooth per-pixel AOD/H2O maps ── */
    float *aod_map_data = NULL;
    float *h2o_map_data = NULL;
    int    have_aod_map = 0;
    int    have_h2o_map = 0;

    if (iso && iso->aod_map && iso->aod_map[0]) {
        G_message(_("Loading per-pixel AOD map <%s>..."), iso->aod_map);
        aod_map_data = load_raster2d(iso->aod_map, nrows, ncols, aod_val);
        /* Clamp to valid AOD range */
        for (int i = 0; i < npix; i++) {
            if (aod_map_data[i] < 0.001f) aod_map_data[i] = 0.001f;
            if (aod_map_data[i] > 5.0f)   aod_map_data[i] = 5.0f;
        }
        if (iso->smooth_sigma > 0.0f) {
            G_message(_("  Gaussian smoothing AOD map (σ=%.1f px)..."),
                      iso->smooth_sigma);
            spatial_gaussian_filter(aod_map_data, nrows, ncols, iso->smooth_sigma);
        }
        have_aod_map = 1;
    }

    if (iso && iso->h2o_map && iso->h2o_map[0]) {
        G_message(_("Loading per-pixel H2O map <%s>..."), iso->h2o_map);
        h2o_map_data = load_raster2d(iso->h2o_map, nrows, ncols, h2o_val);
        for (int i = 0; i < npix; i++) {
            if (h2o_map_data[i] < 0.01f) h2o_map_data[i] = 0.01f;
            if (h2o_map_data[i] > 8.0f)  h2o_map_data[i] = 8.0f;
        }
        if (iso->smooth_sigma > 0.0f) {
            G_message(_("  Gaussian smoothing H2O map (σ=%.1f px)..."),
                      iso->smooth_sigma);
            spatial_gaussian_filter(h2o_map_data, nrows, ncols, iso->smooth_sigma);
        }
        have_h2o_map = 1;
    }

    int use_per_pixel = have_aod_map || have_h2o_map;

    /* ── Pre-compute scalar LUT slice (used when no per-pixel maps) ── */
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
    float  d2f     = (float)d2;
    double cos_sza = cos(sza_deg * (M_PI / 180.0));
    float  cos_szaf = (float)cos_sza;

    /* ── #2: Auto-detect pixel size for adjacency correction ── */
    float pixel_size_m = iso ? iso->pixel_size_m : 0.0f;
    if (iso && iso->adj_psf_km > 0.0f && pixel_size_m <= 0.0f) {
        struct Cell_head win;
        G_get_window(&win);
        pixel_size_m = (float)((win.ew_res + win.ns_res) / 2.0);
        G_verbose_message(_("Auto-detected pixel size: %.1f m"), pixel_size_m);
    }

    /* ── Open output map ── */
    RASTER3D_Map *outmap = Rast3d_open_new_opt_tile_size(
        output_name, RASTER3D_USE_CACHE_X, &region, DCELL_TYPE, 32);
    if (!outmap)
        G_fatal_error(_("Cannot create 3D raster <%s>"), output_name);
    Rast3d_min_unlocked(outmap, RASTER3D_USE_CACHE_X);

    /* ── Open uncertainty output map if requested ── */
    RASTER3D_Map *uncmap = NULL;
    if (iso && iso->do_uncertainty && iso->unc_output && iso->unc_output[0]) {
        uncmap = Rast3d_open_new_opt_tile_size(
            iso->unc_output, RASTER3D_USE_CACHE_X, &region, DCELL_TYPE, 32);
        if (!uncmap)
            G_fatal_error(_("Cannot create uncertainty map <%s>"),
                          iso->unc_output);
        Rast3d_min_unlocked(uncmap, RASTER3D_USE_CACHE_X);
    }

    double null_d;
    Rast_set_d_null_value(&null_d, 1);

    /* ── Allocate working buffers ── */
    /* Per-band buffers (reused each iteration) */
    float *rad_band  = G_malloc((size_t)npix * sizeof(float));
    float *refl_band = G_malloc((size_t)npix * sizeof(float));
    float *sigma_band = (iso && iso->do_uncertainty)
                        ? G_malloc((size_t)npix * sizeof(float)) : NULL;

    /* Full cube needed for MAP regularisation (#3/#5) */
    float *refl_cube  = NULL;
    float *sigma_cube = NULL;

    if (iso && iso->do_surface_prior) {
        refl_cube = G_malloc((size_t)ndepths * npix * sizeof(float));
        if (iso->do_uncertainty)
            sigma_cube = G_malloc((size_t)ndepths * npix * sizeof(float));
        /* Initialise to NaN so unprocessed bands stay NaN */
        for (size_t i = 0; i < (size_t)ndepths * npix; i++) refl_cube[i] = NAN;
        if (sigma_cube)
            for (size_t i = 0; i < (size_t)ndepths * npix; i++) sigma_cube[i] = NAN;
    }

    /* ── #3/#5: Initialise surface model ── */
    SurfaceModelImpl *surf_mdl = NULL;
    float *sigma_model = NULL;   /* per-band model discrepancy */

    if (iso && iso->do_surface_prior) {
        float *wl_bands = G_malloc(ndepths * sizeof(float));
        for (int b = 0; b < ndepths; b++) wl_bands[b] = band_wl[b];
        surf_mdl = surface_model_alloc(wl_bands, ndepths);
        sigma_model = G_malloc(ndepths * sizeof(float));
        if (surf_mdl && sigma_model)
            surface_model_discrepancy(wl_bands, ndepths, sigma_model);
        G_free(wl_bands);
    }

    /* ── Correction loop: band by band ── */
    G_message(_("Correcting %d bands × %d rows × %d cols..."),
              ndepths, nrows, ncols);

    for (int z = 0; z < ndepths; z++) {
        G_percent(z, ndepths, 2);

        float wl  = band_wl[z];
        float E0  = sixs_E0(wl);

        /* Scalar LUT values for this wavelength (fallback / adjacency) */
        float R_a_s = interp_wl(Rs,  cfg->wl, n_wl, wl);
        float T_d_s = interp_wl(Tds, cfg->wl, n_wl, wl);
        float T_u_s = interp_wl(Tus, cfg->wl, n_wl, wl);
        float s_a_s = interp_wl(ss,  cfg->wl, n_wl, wl);

        /* ── Load radiance band ── */
        for (int row = 0; row < nrows; row++)
            for (int col = 0; col < ncols; col++) {
                double L = Rast3d_get_double(inmap, col, row, z);
                rad_band[row * ncols + col] =
                    (Rast_is_d_null_value(&L) || L <= 0.0) ? NAN : (float)L;
            }

        /* ── Per-pixel inversion ── */
        if (use_per_pixel) {
            /* Per-pixel AOD/H2O: trilinear LUT interpolation per pixel */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < npix; i++) {
                float L = rad_band[i];
                if (!isfinite(L) || E0 <= 0.0f) { refl_band[i] = NAN; continue; }

                float a_px = have_aod_map ? aod_map_data[i] : aod_val;
                float h_px = have_h2o_map ? h2o_map_data[i] : h2o_val;

                float R_a, T_d, T_u, s_a;
                atcorr_lut_interp_pixel(cfg, lut, a_px, h_px, wl,
                                         &R_a, &T_d, &T_u, &s_a);

                float rho_toa = (float)(M_PI * L * d2) / (E0 * cos_szaf);
                refl_band[i]  = atcorr_invert(rho_toa, R_a, T_d, T_u, s_a);
            }
        } else {
            /* Scalar correction (original path) */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < npix; i++) {
                float L = rad_band[i];
                if (!isfinite(L) || E0 <= 0.0f) { refl_band[i] = NAN; continue; }
                float rho_toa = (float)(M_PI * L * d2) / (E0 * cos_szaf);
                refl_band[i]  = atcorr_invert(rho_toa, R_a_s, T_d_s, T_u_s, s_a_s);
            }
        }

        /* Clip valid values; NaN stays NaN */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < npix; i++) {
            float r = refl_band[i];
            if (isfinite(r))
                refl_band[i] = (r < -0.01f) ? -0.01f
                             : (r >  1.5f)  ?  1.5f : r;
        }

        /* ── #2: In-loop adjacency correction ── */
        if (iso && iso->adj_psf_km > 0.0f && pixel_size_m > 0.0f) {
            /* Use scene-average AOD for T_dir (good approximation) */
            float aod_mean = aod_val;
            if (have_aod_map) {
                double s = 0.0; int n = 0;
                for (int i = 0; i < npix; i++)
                    if (isfinite(aod_map_data[i])) { s += aod_map_data[i]; n++; }
                if (n > 0) aod_mean = (float)(s / n);
            }
            /* T_scat: use scalar slice (adequate for diffuse fraction) */
            float T_scat = T_d_s * T_u_s;
            adjacency_correct_band(refl_band, nrows, ncols,
                                   iso->adj_psf_km, pixel_size_m,
                                   T_scat, s_a_s,
                                   wl, aod_mean,
                                   cfg->surface_pressure > 0.0f
                                       ? cfg->surface_pressure : 1013.25f,
                                   sza_deg, cfg->vza);
        }

        /* ── #4: Per-band uncertainty ── */
        if (iso && iso->do_uncertainty && sigma_band) {
            uncertainty_compute_band(
                rad_band, refl_band, npix,
                E0, d2f, cos_szaf,
                T_d_s, T_u_s, s_a_s, R_a_s,
                0.0f,  /* nedl = 0 → estimate from rad_band */
                0.04f, /* aod_sigma */
                cfg, lut, wl, aod_val, h2o_val,
                sigma_band);
        }

        /* ── Store or write results ── */
        if (refl_cube) {
            /* Deferred write: accumulate for MAP regularisation */
            memcpy(refl_cube  + (size_t)z * npix, refl_band,
                   (size_t)npix * sizeof(float));
            if (sigma_cube && sigma_band)
                memcpy(sigma_cube + (size_t)z * npix, sigma_band,
                       (size_t)npix * sizeof(float));
        } else {
            /* Immediate write */
            write_band_to_rast3d(outmap, refl_band, nrows, ncols, z, null_d);
            if (uncmap && sigma_band)
                write_band_to_rast3d(uncmap, sigma_band, nrows, ncols, z, null_d);
        }
    }
    G_percent(1, 1, 1);

    /* ── #3/#5: Surface prior MAP regularisation ── */
    if (refl_cube && surf_mdl) {
        G_message(_("Applying surface prior MAP regularisation..."));

        /* If uncertainty available, add model discrepancy in quadrature (#6) */
        float *sigma2_full = NULL;
        if (sigma_cube && sigma_model) {
            sigma2_full = G_malloc((size_t)ndepths * npix * sizeof(float));
            for (int b = 0; b < ndepths; b++) {
                float sd_mdl = sigma_model[b];
                float *sc = sigma_cube + (size_t)b * npix;
                float *s2 = sigma2_full + (size_t)b * npix;
                for (int i = 0; i < npix; i++) {
                    float sc_i = sc[i];
                    if (!isfinite(sc_i)) { s2[i] = NAN; continue; }
                    s2[i] = sc_i * sc_i + sd_mdl * sd_mdl;
                }
            }
        }

        surface_model_regularize(surf_mdl, refl_cube, sigma2_full,
                                  ndepths, npix, 0.1f);

        G_free(sigma2_full);
        G_message(_("  MAP regularisation complete."));
    }

    /* ── Write cube to output (if deferred) ── */
    if (refl_cube) {
        G_message(_("Writing reflectance bands..."));
        for (int z = 0; z < ndepths; z++)
            write_band_to_rast3d(outmap,
                                  refl_cube + (size_t)z * npix,
                                  nrows, ncols, z, null_d);

        if (uncmap && sigma_cube) {
            G_message(_("Writing uncertainty bands..."));
            for (int z = 0; z < ndepths; z++)
                write_band_to_rast3d(uncmap,
                                      sigma_cube + (size_t)z * npix,
                                      nrows, ncols, z, null_d);
        }
    }

    /* ── Close maps ── */
    if (!Rast3d_close(inmap))
        G_fatal_error(_("Cannot close input map <%s>"), input_name);
    if (!Rast3d_close(outmap))
        G_fatal_error(_("Cannot close output map <%s>"), output_name);
    if (uncmap && !Rast3d_close(uncmap))
        G_fatal_error(_("Cannot close uncertainty map <%s>"), iso->unc_output);

    /* ── Cleanup ── */
    G_free(band_wl);    G_free(band_fwhm);
    G_free(Rs);         G_free(Tds);     G_free(Tus);   G_free(ss);
    G_free(rad_band);   G_free(refl_band);
    G_free(sigma_band);
    G_free(refl_cube);  G_free(sigma_cube);
    G_free(aod_map_data);
    G_free(h2o_map_data);
    G_free(sigma_model);
    surface_model_free(surf_mdl);

    G_message(_("Surface reflectance written to <%s>"), output_name);
    if (uncmap)
        G_message(_("Uncertainty written to <%s>"), iso->unc_output);
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
    opt_sza->key = "sza"; opt_sza->type = TYPE_DOUBLE; opt_sza->required = YES;
    opt_sza->label = _("Solar zenith angle (degrees, 0–89)");
    opt_sza->guisection = _("Geometry");

    struct Option *opt_vza = G_define_option();
    opt_vza->key = "vza"; opt_vza->type = TYPE_DOUBLE;
    opt_vza->required = NO; opt_vza->answer = "0";
    opt_vza->label = _("View zenith angle (degrees, 0–60)");
    opt_vza->guisection = _("Geometry");

    struct Option *opt_raa = G_define_option();
    opt_raa->key = "raa"; opt_raa->type = TYPE_DOUBLE;
    opt_raa->required = NO; opt_raa->answer = "0";
    opt_raa->label = _("Relative azimuth angle (degrees)");
    opt_raa->guisection = _("Geometry");

    struct Option *opt_altitude = G_define_option();
    opt_altitude->key = "altitude"; opt_altitude->type = TYPE_DOUBLE;
    opt_altitude->required = NO; opt_altitude->answer = "1000";
    opt_altitude->label = _("Sensor altitude (km; > 900 = satellite)");
    opt_altitude->guisection = _("Geometry");

    /* ── Atmosphere ── */
    struct Option *opt_atmo = G_define_option();
    opt_atmo->key = "atmosphere"; opt_atmo->type = TYPE_STRING;
    opt_atmo->required = NO; opt_atmo->answer = "us62";
    opt_atmo->options = "us62,midsum,midwin,tropical,subsum,subwin";
    opt_atmo->description = _("Standard atmosphere model");
    opt_atmo->guisection = _("Atmosphere");

    struct Option *opt_aerosol = G_define_option();
    opt_aerosol->key = "aerosol"; opt_aerosol->type = TYPE_STRING;
    opt_aerosol->required = NO; opt_aerosol->answer = "continental";
    opt_aerosol->options = "none,continental,maritime,urban,desert";
    opt_aerosol->description = _("Aerosol model");
    opt_aerosol->guisection = _("Atmosphere");

    struct Option *opt_ozone = G_define_option();
    opt_ozone->key = "ozone"; opt_ozone->type = TYPE_DOUBLE;
    opt_ozone->required = NO; opt_ozone->answer = "300";
    opt_ozone->description = _("Total ozone column (Dobson units)");
    opt_ozone->guisection = _("Atmosphere");

    /* ── LUT grid ── */
    struct Option *opt_aod = G_define_option();
    opt_aod->key = "aod"; opt_aod->type = TYPE_STRING;
    opt_aod->required = NO; opt_aod->answer = "0.0,0.05,0.1,0.2,0.4,0.8";
    opt_aod->label = _("AOD at 550 nm grid (comma-separated)");
    opt_aod->guisection = _("LUT");

    struct Option *opt_h2o = G_define_option();
    opt_h2o->key = "h2o"; opt_h2o->type = TYPE_STRING;
    opt_h2o->required = NO; opt_h2o->answer = "0.5,1.0,2.0,3.5,5.0";
    opt_h2o->label = _("Column water vapour grid in g/cm² (comma-separated)");
    opt_h2o->guisection = _("LUT");

    struct Option *opt_wl_min = G_define_option();
    opt_wl_min->key = "wl_min"; opt_wl_min->type = TYPE_DOUBLE;
    opt_wl_min->required = NO; opt_wl_min->answer = "0.40";
    opt_wl_min->description = _("Minimum wavelength (µm)");
    opt_wl_min->guisection = _("LUT");

    struct Option *opt_wl_max = G_define_option();
    opt_wl_max->key = "wl_max"; opt_wl_max->type = TYPE_DOUBLE;
    opt_wl_max->required = NO; opt_wl_max->answer = "2.50";
    opt_wl_max->description = _("Maximum wavelength (µm)");
    opt_wl_max->guisection = _("LUT");

    struct Option *opt_wl_step = G_define_option();
    opt_wl_step->key = "wl_step"; opt_wl_step->type = TYPE_DOUBLE;
    opt_wl_step->required = NO; opt_wl_step->answer = "0.01";
    opt_wl_step->description = _("Wavelength step (µm)");
    opt_wl_step->guisection = _("LUT");

    /* ── Correction parameters ── */
    struct Option *opt_doy = G_define_option();
    opt_doy->key = "doy"; opt_doy->type = TYPE_INTEGER;
    opt_doy->required = NO; opt_doy->answer = "180";
    opt_doy->label = _("Day of year for Earth-Sun distance (1–365)");
    opt_doy->guisection = _("Correction");

    struct Option *opt_aod_val = G_define_option();
    opt_aod_val->key = "aod_val"; opt_aod_val->type = TYPE_DOUBLE;
    opt_aod_val->required = NO; opt_aod_val->answer = "0.1";
    opt_aod_val->label = _("Scene AOD at 550 nm for correction (scalar fallback)");
    opt_aod_val->guisection = _("Correction");

    struct Option *opt_h2o_val = G_define_option();
    opt_h2o_val->key = "h2o_val"; opt_h2o_val->type = TYPE_DOUBLE;
    opt_h2o_val->required = NO; opt_h2o_val->answer = "2.0";
    opt_h2o_val->label = _("Scene column water vapour g/cm² (scalar fallback)");
    opt_h2o_val->guisection = _("Correction");

    /* ── ISOFIT improvements ── */

    /* #1 Per-pixel maps */
    struct Option *opt_aod_map = G_define_standard_option(G_OPT_R_INPUT);
    opt_aod_map->key = "aod_map"; opt_aod_map->required = NO;
    opt_aod_map->label = _("Per-pixel AOD raster (overrides aod_val= where non-null)");
    opt_aod_map->description = _("2-D raster of AOD at 550 nm");
    opt_aod_map->guisection = _("ISOFIT");

    struct Option *opt_h2o_map = G_define_standard_option(G_OPT_R_INPUT);
    opt_h2o_map->key = "h2o_map"; opt_h2o_map->required = NO;
    opt_h2o_map->label = _("Per-pixel water vapour raster (overrides h2o_val= where non-null)");
    opt_h2o_map->description = _("2-D raster of column water vapour in g/cm²");
    opt_h2o_map->guisection = _("ISOFIT");

    struct Option *opt_smooth = G_define_option();
    opt_smooth->key = "smooth"; opt_smooth->type = TYPE_DOUBLE;
    opt_smooth->required = NO; opt_smooth->answer = "0";
    opt_smooth->label = _("Gaussian smoothing σ in pixels for AOD/H2O maps");
    opt_smooth->description =
        _("Spatially smooths aod_map= and h2o_map= before correction. "
          "0 = disabled. Typical value: 2–5 pixels.");
    opt_smooth->guisection = _("ISOFIT");

    /* #2 Adjacency */
    struct Option *opt_adj_psf = G_define_option();
    opt_adj_psf->key = "adj_psf"; opt_adj_psf->type = TYPE_DOUBLE;
    opt_adj_psf->required = NO; opt_adj_psf->answer = "0";
    opt_adj_psf->label = _("Adjacency effect PSF radius (km; 0 = disabled)");
    opt_adj_psf->description =
        _("Environmental PSF radius for Vermote 1997 adjacency correction. "
          "Typical value: 0.5–2 km. 0 = disabled.");
    opt_adj_psf->guisection = _("ISOFIT");

    struct Option *opt_pixel_size = G_define_option();
    opt_pixel_size->key = "pixel_size"; opt_pixel_size->type = TYPE_DOUBLE;
    opt_pixel_size->required = NO; opt_pixel_size->answer = "0";
    opt_pixel_size->label = _("Pixel size in metres (0 = auto-detect from region)");
    opt_pixel_size->guisection = _("ISOFIT");

    /* #4/#6 Uncertainty output */
    struct Option *opt_uncertainty;
    opt_uncertainty = G_define_standard_option(G_OPT_R3_OUTPUT);
    opt_uncertainty->key = "uncertainty"; opt_uncertainty->required = NO;
    opt_uncertainty->label = _("Output uncertainty Raster3D map (requires -u flag)");
    opt_uncertainty->guisection = _("ISOFIT");

    /* Flags */
    struct Flag *flag_u = G_define_flag();
    flag_u->key = 'u';
    flag_u->label = _("Compute per-band reflectance uncertainty");
    flag_u->description =
        _("Propagates instrument noise and AOD uncertainty through the "
          "inversion. Stores σ_rfl per pixel per band. "
          "Write to Raster3D with uncertainty= option.");
    flag_u->guisection = _("ISOFIT");

    struct Flag *flag_r = G_define_flag();
    flag_r->key = 'r';
    flag_r->label = _("Apply surface prior MAP regularisation");
    flag_r->description =
        _("After inverting all bands, blends retrieved reflectance with a "
          "3-component Gaussian mixture surface prior (vegetation/soil/water) "
          "using diagonal MAP estimation. Requires loading the full cube into "
          "memory. Uses per-band model discrepancy as uncertainty floor (#6).");
    flag_r->guisection = _("ISOFIT");

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    /* ── Validate mode ── */
    if (!opt_lut->answer && !opt_output->answer)
        G_fatal_error(_("Specify at least one output: lut= or output="));
    if (opt_output->answer && !opt_input->answer)
        G_fatal_error(_("output= requires input="));
    if (opt_uncertainty->answer && !flag_u->answer)
        G_warning(_("uncertainty= output ignored without -u flag"));
    if (flag_u->answer && !opt_output->answer)
        G_fatal_error(_("-u requires output= (needs correction to run first)"));

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
    if      (!strcmp(opt_atmo->answer, "us62"))     atmo_model = ATMO_US62;
    else if (!strcmp(opt_atmo->answer, "midsum"))   atmo_model = ATMO_MIDSUM;
    else if (!strcmp(opt_atmo->answer, "midwin"))   atmo_model = ATMO_MIDWIN;
    else if (!strcmp(opt_atmo->answer, "tropical")) atmo_model = ATMO_TROPICAL;
    else if (!strcmp(opt_atmo->answer, "subsum"))   atmo_model = ATMO_SUBSUM;
    else if (!strcmp(opt_atmo->answer, "subwin"))   atmo_model = ATMO_SUBWIN;
    else G_fatal_error(_("Unknown atmosphere model: %s"), opt_atmo->answer);

    int aerosol_model;
    if      (!strcmp(opt_aerosol->answer, "none"))        aerosol_model = AEROSOL_NONE;
    else if (!strcmp(opt_aerosol->answer, "continental")) aerosol_model = AEROSOL_CONTINENTAL;
    else if (!strcmp(opt_aerosol->answer, "maritime"))    aerosol_model = AEROSOL_MARITIME;
    else if (!strcmp(opt_aerosol->answer, "urban"))       aerosol_model = AEROSOL_URBAN;
    else if (!strcmp(opt_aerosol->answer, "desert"))      aerosol_model = AEROSOL_DESERT;
    else G_fatal_error(_("Unknown aerosol model: %s"), opt_aerosol->answer);

    float aod_buf[64]; int n_aod = parse_csv_floats(opt_aod->answer, aod_buf, 64);
    if (n_aod <= 0) G_fatal_error(_("Cannot parse aod= values"));

    float h2o_buf[64]; int n_h2o = parse_csv_floats(opt_h2o->answer, h2o_buf, 64);
    if (n_h2o <= 0) G_fatal_error(_("Cannot parse h2o= values"));

    int n_wl = (int)((wl_max - wl_min) / wl_step) + 1;
    if (n_wl < 1 || n_wl > 10000)
        G_fatal_error(_("Wavelength grid size %d out of valid range"), n_wl);

    float *wl_buf = G_malloc(n_wl * sizeof(float));
    for (int i = 0; i < n_wl; i++) wl_buf[i] = wl_min + i * wl_step;

    /* ── Build LUT config ── */
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
        /* Log enabled improvements */
        if (opt_aod_map->answer || opt_h2o_map->answer) {
            G_message(_("Per-pixel atmospheric maps enabled:"));
            if (opt_aod_map->answer)
                G_message(_("  AOD map: <%s>"), opt_aod_map->answer);
            if (opt_h2o_map->answer)
                G_message(_("  H2O map: <%s>"), opt_h2o_map->answer);
            float s = (float)atof(opt_smooth->answer);
            if (s > 0.0f) G_message(_("  Gaussian smooth σ=%.1f px"), s);
        }
        float adj = (float)atof(opt_adj_psf->answer);
        if (adj > 0.0f)
            G_message(_("Adjacency correction: PSF=%.2f km"), adj);
        if (flag_r->answer)
            G_message(_("Surface prior MAP regularisation: ENABLED"));
        if (flag_u->answer)
            G_message(_("Reflectance uncertainty: ENABLED"));

        IsoFitParams iso = {
            .aod_map        = opt_aod_map->answer,
            .h2o_map        = opt_h2o_map->answer,
            .smooth_sigma   = (float)atof(opt_smooth->answer),
            .adj_psf_km     = adj,
            .pixel_size_m   = (float)atof(opt_pixel_size->answer),
            .do_surface_prior = flag_r->answer ? 1 : 0,
            .do_uncertainty = flag_u->answer ? 1 : 0,
            .unc_output     = (flag_u->answer && opt_uncertainty->answer)
                              ? opt_uncertainty->answer : NULL,
        };

        correct_raster3d(opt_input->answer, opt_output->answer,
                         &cfg, &lut, aod_val, h2o_val, doy, sza, &iso);
    }

    G_free(wl_buf);
    G_free(R_atm); G_free(T_down); G_free(T_up); G_free(s_alb);

    exit(EXIT_SUCCESS);
}
