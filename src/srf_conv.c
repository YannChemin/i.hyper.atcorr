/* srf_conv.c — Spectral Response Function (SRF) convolution for gas transmittance.
 *
 * For hyperspectral sensors with FWHM < 5 nm, the 6SV Curtis-Godson
 * parameterisation (10 cm⁻¹ = ~0.5–2 nm) introduces errors in gas absorption
 * bands (H₂O at 720/820/940/1380 nm, O₂-A at 760 nm, CO₂ at 1600/2000 nm).
 * Errors reach 20–40% transmittance at 1 nm FWHM, and up to 80% at the O₂-A
 * band for sensors narrower than 0.5 nm.
 *
 * Correction method:
 *   1. Call libRadtran uvspec with mol_abs_param reptran fine (~0.05 nm)
 *      for solar and viewing paths.  T_eff(λ) = edir_surface / edir_toa.
 *      (Rayleigh scattering is present but cancels in the fine/coarse ratio
 *      because it is spectrally smooth over any sensor FWHM.)
 *   2. Convolve T_gas_fine with the sensor Gaussian SRF per band.
 *   3. Repeat with reptran coarse — this matches the resolution the LUT was
 *      built at (6SV Curtis-Godson at ~10 cm⁻¹).
 *   4. Correction = T_gas_fine_SRF / T_gas_coarse_at_band_centre
 *   5. Apply multiplicatively to all T_down and T_up in the LUT.
 *      T_scat is smooth spectrally; only T_gas needs SRF correction.
 *
 * OpenMP: 4 × n_h2o uvspec calls are run in parallel (one call per
 * {resolution, direction, h2o} triplet).  Each thread uses its own temp
 * directory so subprocesses never collide.
 *
 * libRadtran dependency: runtime only (subprocess via popen).  No link-time
 * dependency.  Set LIBRADTRAN_DATA to override default data path.
 */

#define _GNU_SOURCE
#include "../include/atcorr.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <limits.h>
#include <omp.h>

/* ─── Constants ───────────────────────────────────────────────────────────── */

#define SRF_WINDOW_SIGMA   4.0f    /* Gaussian truncation: ±4σ → 99.994% */
#define SRF_MAX_PTS       300000   /* max reptran fine output wavelengths  */
#define SRF_MARGIN_NM      15.0f   /* spectral margin beyond sensor range  */

/* ─── Internal opaque struct ─────────────────────────────────────────────── */

struct SrfCorrection_ {
    float *corr_down;   /* [n_wl * n_h2o] — index iw * n_h2o + ih */
    float *corr_up;
    int    n_wl;
    int    n_h2o;
};

/* ─── Gas spectrum from one uvspec run ───────────────────────────────────── */

typedef struct {
    float *wl;      /* nm [n_pts] — sorted ascending */
    float *T_gas;   /* transmittance [n_pts] */
    int    n_pts;
} GasSpectrum;

static void gas_spectrum_free(GasSpectrum *g)
{
    if (!g) return;
    free(g->wl);
    free(g->T_gas);
    free(g);
}

/* ─── libRadtran path discovery ─────────────────────────────────────────── */

static int find_uvspec(char *buf, size_t len)
{
    static const char *cands[] = {
        "/usr/local/bin/uvspec",
        "/usr/bin/uvspec",
        NULL
    };

    /* GISBASE/../bin/uvspec (installed alongside GRASS) */
    const char *gb = getenv("GISBASE");
    if (gb) {
        snprintf(buf, len, "%s/../bin/uvspec", gb);
        if (access(buf, X_OK) == 0) return 0;
        snprintf(buf, len, "%s/bin/uvspec", gb);
        if (access(buf, X_OK) == 0) return 0;
    }

    /* LIBRADTRAN_DIR env override */
    const char *lrd = getenv("LIBRADTRAN_DIR");
    if (lrd) {
        snprintf(buf, len, "%s/bin/uvspec", lrd);
        if (access(buf, X_OK) == 0) return 0;
    }

    for (int i = 0; cands[i]; i++) {
        if (access(cands[i], X_OK) == 0) {
            snprintf(buf, len, "%s", cands[i]);
            return 0;
        }
    }
    return -1;
}

static int find_data_path(char *buf, size_t len)
{
    static const char *cands[] = {
        "/usr/local/share/libRadtran/data",
        "/usr/share/libRadtran/data",
        NULL
    };

    /* Explicit override */
    const char *lrdata = getenv("LIBRADTRAN_DATA");
    if (lrdata) {
        char test[PATH_MAX];
        snprintf(test, sizeof(test), "%s/atmmod/afglus.dat", lrdata);
        if (access(test, R_OK) == 0) {
            snprintf(buf, len, "%s", lrdata);
            return 0;
        }
    }

    for (int i = 0; cands[i]; i++) {
        char test[PATH_MAX];
        snprintf(test, sizeof(test), "%s/atmmod/afglus.dat", cands[i]);
        if (access(test, R_OK) == 0) {
            snprintf(buf, len, "%s", cands[i]);
            return 0;
        }
    }
    return -1;
}

/* ─── Gaussian SRF convolution ──────────────────────────────────────────── */

/* Convolve spectrum (wl[], T[], n) with Gaussian SRF centred at centre_nm
 * with given fwhm_nm.  Weights: w = exp(-0.5*(Δλ/σ)²), σ = fwhm/(2√(2ln2)).
 * Returns SRF-weighted mean transmittance; falls back to nearest if no points
 * fall within the ±4σ window. */
static float gaussian_srf_convolve(const float *wl, const float *T, int n,
                                    float centre_nm, float fwhm_nm)
{
    float sigma    = fwhm_nm / (2.0f * sqrtf(2.0f * logf(2.0f)));
    float half_win = SRF_WINDOW_SIGMA * sigma;
    float lo       = centre_nm - half_win;
    float hi       = centre_nm + half_win;

    /* Binary search for start of window */
    int l = 0, r = n - 1;
    while (l < r) {
        int m = (l + r) / 2;
        if (wl[m] < lo) l = m + 1; else r = m;
    }

    double sum_w = 0.0, sum_wT = 0.0;
    for (int i = l; i < n && wl[i] <= hi; i++) {
        double d = (double)(wl[i] - centre_nm) / (double)sigma;
        double w = exp(-0.5 * d * d);
        sum_w  += w;
        sum_wT += w * (double)T[i];
    }

    if (sum_w < 1e-30) {
        /* No points in window — return nearest */
        int nearest = 0;
        float best = fabsf(wl[0] - centre_nm);
        for (int i = 1; i < n; i++) {
            float dist = fabsf(wl[i] - centre_nm);
            if (dist < best) { best = dist; nearest = i; }
        }
        return T[nearest];
    }
    float result = (float)(sum_wT / sum_w);
    if (result < 0.0f) result = 0.0f;
    if (result > 1.0f) result = 1.0f;
    return result;
}

/* ─── Linear interpolation ──────────────────────────────────────────────── */

static float interp1d_nm(const float *x, const float *y, int n, float xi)
{
    if (n <= 0)            return 1.0f;
    if (xi <= x[0])        return y[0];
    if (xi >= x[n - 1])    return y[n - 1];

    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (x[mid] <= xi) lo = mid; else hi = mid;
    }
    float t = (xi - x[lo]) / (x[hi] - x[lo]);
    float v = y[lo] + t * (y[hi] - y[lo]);
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return v;
}

/* ─── Run uvspec for gas-only transmittance ─────────────────────────────── */

/* Runs uvspec with mol_abs_param reptran {resolution}, no_rayleigh, no
 * aerosol. Returns T_gas = edir_surface / edir_toa.
 *
 * sza_deg: zenith angle of the beam path (solar for down, view for up).
 * h2o_gcm2: water vapour column (g/cm²).
 * o3_du: ozone column (Dobson units).
 * wl_min_nm, wl_max_nm: spectral range requested.
 * resolution: "fine" (~0.05 nm) or "coarse" (~0.16 nm).
 * thread_id: used to make temp-dir name unique per OMP thread.
 *
 * Returns GasSpectrum* on success (caller frees), NULL on error. */
static GasSpectrum *run_uvspec_gas(float sza_deg, float h2o_gcm2, float o3_du,
                                    float wl_min_nm, float wl_max_nm,
                                    const char *resolution,
                                    const char *uvspec_path,
                                    const char *data_path,
                                    int thread_id)
{
    /* ── Create temp directory ── */
    char tmpdir[PATH_MAX];
    snprintf(tmpdir, sizeof(tmpdir), "/tmp/atcorr_srf_%d_XXXXXX", thread_id);
    if (!mkdtemp(tmpdir)) return NULL;

    char inp_path[PATH_MAX + 16];  /* +16 for "/uvspec.inp\0" */
    snprintf(inp_path, sizeof(inp_path), "%s/uvspec.inp", tmpdir);

    /* ── Write uvspec input ── */
    FILE *f = fopen(inp_path, "w");
    if (!f) { rmdir(tmpdir); return NULL; }

    /* mol_modify H2O takes precipitable water in mm (1 g/cm² = 10 mm) */
    float h2o_mm  = h2o_gcm2 * 10.0f;
    float wl_lo   = wl_min_nm - (float)SRF_MARGIN_NM;
    float wl_hi   = wl_max_nm + (float)SRF_MARGIN_NM;
    if (wl_lo < 200.0f) wl_lo = 200.0f;
    if (wl_hi > 4900.0f) wl_hi = 4900.0f;

    /* Clamp SZA: disort fails at 90° */
    if (sza_deg > 89.0f) sza_deg = 89.0f;
    if (sza_deg < 0.0f)  sza_deg = 0.0f;

    fprintf(f,
        "data_files_path %s\n"
        "atmosphere_file %s/atmmod/afglus.dat\n"
        "source solar %s/solar_flux/kurudz_1.0nm.dat per_nm\n"
        "wavelength %.2f %.2f\n"
        "mol_abs_param reptran %s\n"
        "mol_modify H2O %.4f MM\n"
        "mol_modify O3 %.2f DU\n"
        "sza %.4f\n"
        "phi0 0.0\n"
        "umu 1.0\n"
        "albedo 0.0\n"
        "pressure 1013.25\n"
        "rte_solver disort\n"
        "number_of_streams 2\n"
        "zout sur toa\n"
        "output_user lambda edir\n"
        "quiet\n",
        data_path, data_path, data_path,
        (double)wl_lo, (double)wl_hi,
        resolution,
        (double)h2o_mm,
        (double)(o3_du > 0.0f ? o3_du : 300.0f),
        (double)sza_deg);
    fclose(f);

    /* ── Run uvspec ── */
    char cmd[PATH_MAX * 2 + 16];
    snprintf(cmd, sizeof(cmd), "%s < %s 2>/dev/null", uvspec_path, inp_path);

    FILE *pipe = popen(cmd, "r");
    if (!pipe) {
        unlink(inp_path); rmdir(tmpdir);
        return NULL;
    }

    /* ── Allocate output ── */
    GasSpectrum *gs = malloc(sizeof(GasSpectrum));
    if (!gs) { pclose(pipe); unlink(inp_path); rmdir(tmpdir); return NULL; }
    gs->wl    = malloc((size_t)SRF_MAX_PTS * sizeof(float));
    gs->T_gas = malloc((size_t)SRF_MAX_PTS * sizeof(float));
    gs->n_pts = 0;
    if (!gs->wl || !gs->T_gas) {
        gas_spectrum_free(gs); pclose(pipe);
        unlink(inp_path); rmdir(tmpdir);
        return NULL;
    }

    /* ── Parse 2-row-per-wavelength output ──
     * zout sur toa → rows alternate: surface (even), TOA (odd). */
    char   line[128];
    int    is_sur    = 1;
    float  edir_sur  = 0.0f;
    float  last_wl   = -1.0f;

    while (fgets(line, sizeof(line), pipe)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        float wl_v, edir_v;
        if (sscanf(line, "%f %f", &wl_v, &edir_v) != 2) continue;

        if (is_sur) {
            edir_sur = edir_v;
            last_wl  = wl_v;
            is_sur   = 0;
        } else {
            /* TOA row: compute T_gas = edir_sur / edir_toa */
            float edir_toa = (edir_v > 1e-30f) ? edir_v : 1e-30f;
            float T = edir_sur / edir_toa;
            if (T < 0.0f) T = 0.0f;
            if (T > 1.0f) T = 1.0f;

            /* Only store points within requested range */
            if (last_wl >= wl_min_nm && last_wl <= wl_max_nm
                    && gs->n_pts < SRF_MAX_PTS) {
                gs->wl[gs->n_pts]    = last_wl;
                gs->T_gas[gs->n_pts] = T;
                gs->n_pts++;
            }
            is_sur = 1;
        }
    }

    pclose(pipe);
    unlink(inp_path);
    rmdir(tmpdir);

    if (gs->n_pts == 0) {
        gas_spectrum_free(gs);
        return NULL;
    }
    return gs;
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

/* Compute SRF correction table.
 *
 * For each h2o grid point, runs 4 uvspec calls:
 *   fine ×  solar zenith → T_gas_down_fine[wl]
 *   fine ×  view  zenith → T_gas_up_fine[wl]
 *   coarse × solar zenith → T_gas_down_coarse[wl]
 *   coarse × view  zenith → T_gas_up_coarse[wl]
 *
 * Correction[iw, ih] = gaussian_convolve(fine[wl]) / interp(coarse, wl[iw])
 *
 * All 4 × n_h2o calls are distributed across OpenMP threads.
 *
 * Returns NULL if uvspec/data not found or all runs failed. */
SrfCorrection *atcorr_srf_compute(const SrfConfig *srf_cfg,
                                   const LutConfig *lut_cfg)
{
    char uvspec_path[PATH_MAX], data_path[PATH_MAX];

    if (find_uvspec(uvspec_path, sizeof(uvspec_path)) != 0) {
        fprintf(stderr, "[srf_conv] uvspec not found — set LIBRADTRAN_DIR\n");
        return NULL;
    }
    if (find_data_path(data_path, sizeof(data_path)) != 0) {
        fprintf(stderr, "[srf_conv] libRadtran data not found — set LIBRADTRAN_DATA\n");
        return NULL;
    }

    int n_wl  = lut_cfg->n_wl;
    int n_h2o = lut_cfg->n_h2o;

    /* Sensor range in nm */
    float wl_min_nm = lut_cfg->wl[0]        * 1000.0f;   /* µm → nm */
    float wl_max_nm = lut_cfg->wl[n_wl - 1] * 1000.0f;

    float o3_du  = (lut_cfg->ozone_du > 0.0f)
                       ? lut_cfg->ozone_du : 300.0f;
    float sza    = lut_cfg->sza;
    float vza    = lut_cfg->vza;
    float thresh = (srf_cfg && srf_cfg->threshold_um > 0.0f)
                       ? srf_cfg->threshold_um : 0.005f; /* 5 nm */

    /* Count bands needing correction */
    int n_to_correct = 0;
    if (srf_cfg && srf_cfg->fwhm_um) {
        for (int iw = 0; iw < n_wl; iw++)
            if (srf_cfg->fwhm_um[iw] < thresh) n_to_correct++;
    } else {
        n_to_correct = n_wl;  /* no FWHM provided: correct all */
    }
    if (n_to_correct == 0) {
        fprintf(stderr, "[srf_conv] no bands below %.1f nm threshold — "
                        "skipping SRF correction\n", thresh * 1000.0f);
        return NULL;
    }
    fprintf(stderr, "[srf_conv] SRF correction: %d/%d bands need reptran fine "
                    "(threshold %.1f nm)\n",
            n_to_correct, n_wl, thresh * 1000.0f);

    /* ── Allocate output ── */
    SrfCorrection *srf = calloc(1, sizeof(SrfCorrection));
    if (!srf) return NULL;
    srf->n_wl  = n_wl;
    srf->n_h2o = n_h2o;
    srf->corr_down = malloc((size_t)n_wl * n_h2o * sizeof(float));
    srf->corr_up   = malloc((size_t)n_wl * n_h2o * sizeof(float));
    if (!srf->corr_down || !srf->corr_up) {
        atcorr_srf_free(srf); return NULL;
    }
    /* Default: no correction */
    for (int i = 0; i < n_wl * n_h2o; i++) {
        srf->corr_down[i] = 1.0f;
        srf->corr_up[i]   = 1.0f;
    }

    /* ── Schedule 4 × n_h2o uvspec runs ──
     * run_id = i_res * 2*n_h2o  +  i_dir * n_h2o  +  ih
     *   i_res : 0 = fine, 1 = coarse
     *   i_dir : 0 = down (sza), 1 = up (vza)
     *   ih    : h2o grid index */
    int total_runs = 4 * n_h2o;
    GasSpectrum **spectra = calloc((size_t)total_runs, sizeof(GasSpectrum *));
    if (!spectra) { atcorr_srf_free(srf); return NULL; }

    #pragma omp parallel for schedule(dynamic)
    for (int run = 0; run < total_runs; run++) {
        int i_res = run / (2 * n_h2o);
        int i_dir = (run % (2 * n_h2o)) / n_h2o;
        int ih    = run % n_h2o;

        const char *res   = (i_res == 0) ? "fine" : "coarse";
        float       sza_r = (i_dir == 0) ? sza : vza;
        float       h2o_v = lut_cfg->h2o[ih];

        int tid = omp_get_thread_num();

        spectra[run] = run_uvspec_gas(
            sza_r, h2o_v, o3_du,
            wl_min_nm, wl_max_nm,
            res, uvspec_path, data_path, tid);

        if (!spectra[run])
            fprintf(stderr, "[srf_conv] run %d failed "
                            "(%s, %s, H2O=%.2f)\n",
                    run, res,
                    i_dir == 0 ? "down" : "up", h2o_v);
    }

    /* ── Compute correction factors ── */
    for (int ih = 0; ih < n_h2o; ih++) {
        int r_fd = 0 * 2 * n_h2o + 0 * n_h2o + ih;   /* fine,   down */
        int r_fu = 0 * 2 * n_h2o + 1 * n_h2o + ih;   /* fine,   up   */
        int r_cd = 1 * 2 * n_h2o + 0 * n_h2o + ih;   /* coarse, down */
        int r_cu = 1 * 2 * n_h2o + 1 * n_h2o + ih;   /* coarse, up   */

        GasSpectrum *fd = spectra[r_fd];
        GasSpectrum *fu = spectra[r_fu];
        GasSpectrum *cd = spectra[r_cd];
        GasSpectrum *cu = spectra[r_cu];

        if (!fd || !fu || !cd || !cu) continue;

        for (int iw = 0; iw < n_wl; iw++) {
            float wl_um   = lut_cfg->wl[iw];
            float wl_nm   = wl_um * 1000.0f;
            float fwhm_nm = (srf_cfg && srf_cfg->fwhm_um)
                                ? srf_cfg->fwhm_um[iw] * 1000.0f
                                : 0.0f;   /* 0 → treat as delta: use nearest */

            /* Skip bands above threshold */
            if (srf_cfg && srf_cfg->fwhm_um
                    && srf_cfg->fwhm_um[iw] >= thresh)
                continue;

            int idx = iw * n_h2o + ih;

            if (fwhm_nm > 0.05f) {
                /* Gaussian SRF convolution of fine spectrum */
                float Tfd_srf = gaussian_srf_convolve(
                    fd->wl, fd->T_gas, fd->n_pts, wl_nm, fwhm_nm);
                float Tfu_srf = gaussian_srf_convolve(
                    fu->wl, fu->T_gas, fu->n_pts, wl_nm, fwhm_nm);
                /* Coarse reference at band centre */
                float Tcd_ref = interp1d_nm(cd->wl, cd->T_gas, cd->n_pts, wl_nm);
                float Tcu_ref = interp1d_nm(cu->wl, cu->T_gas, cu->n_pts, wl_nm);

                srf->corr_down[idx] = (Tcd_ref > 1e-6f) ? Tfd_srf / Tcd_ref : 1.0f;
                srf->corr_up[idx]   = (Tcu_ref > 1e-6f) ? Tfu_srf / Tcu_ref : 1.0f;
            } else {
                /* Delta SRF (or negligible FWHM): use fine at nearest point */
                float Tfd = interp1d_nm(fd->wl, fd->T_gas, fd->n_pts, wl_nm);
                float Tfu = interp1d_nm(fu->wl, fu->T_gas, fu->n_pts, wl_nm);
                float Tcd = interp1d_nm(cd->wl, cd->T_gas, cd->n_pts, wl_nm);
                float Tcu = interp1d_nm(cu->wl, cu->T_gas, cu->n_pts, wl_nm);

                srf->corr_down[idx] = (Tcd > 1e-6f) ? Tfd / Tcd : 1.0f;
                srf->corr_up[idx]   = (Tcu > 1e-6f) ? Tfu / Tcu : 1.0f;
            }

            /* Physical clamp: correction shouldn't be outside [0.1, 3.0].
             * Values outside this range indicate a uvspec or LUT anomaly. */
            srf->corr_down[idx] = fmaxf(0.1f, fminf(3.0f, srf->corr_down[idx]));
            srf->corr_up[idx]   = fmaxf(0.1f, fminf(3.0f, srf->corr_up[idx]));
        }
    }

    /* ── Cleanup ── */
    for (int run = 0; run < total_runs; run++)
        gas_spectrum_free(spectra[run]);
    free(spectra);

    return srf;
}

/* Apply SRF correction multiplicatively to all T_down and T_up in the LUT.
 * Must be called before pixel-level inversion.
 * LUT arrays are modified in place: T_down[*] *= corr_down, T_up[*] *= corr_up. */
void atcorr_srf_apply(const SrfCorrection *srf,
                       const LutConfig *cfg, LutArrays *lut)
{
    if (!srf || !cfg || !lut) return;

    int n_aod = cfg->n_aod;
    int n_h2o = cfg->n_h2o;
    int n_wl  = cfg->n_wl;

    for (int ia = 0; ia < n_aod; ia++) {
        for (int ih = 0; ih < n_h2o; ih++) {
            for (int iw = 0; iw < n_wl; iw++) {
                size_t lut_idx = (size_t)ia * n_h2o * n_wl
                               + (size_t)ih * n_wl
                               + (size_t)iw;
                int    srf_idx = iw * n_h2o + ih;

                float td = lut->T_down[lut_idx] * srf->corr_down[srf_idx];
                float tu = lut->T_up[lut_idx]   * srf->corr_up[srf_idx];

                lut->T_down[lut_idx] = fmaxf(0.0f, fminf(1.0f, td));
                lut->T_up[lut_idx]   = fmaxf(0.0f, fminf(1.0f, tu));
            }
        }
    }
}

void atcorr_srf_free(SrfCorrection *srf)
{
    if (!srf) return;
    free(srf->corr_down);
    free(srf->corr_up);
    free(srf);
}
