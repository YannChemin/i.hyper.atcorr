/**
 * \file spectral_brdf.c
 * \brief MCD43 disaggregation and Tikhonov spectral smoothing.
 *
 * Implements:
 *   - spectral_interp_linear(): piecewise-linear interpolation with
 *     clamp-extrapolation outside the anchor range.
 *   - spectral_smooth_tikhonov(): O(5n) Cholesky-band solver for the
 *     second-difference Tikhonov regularization system.
 *   - mcd43_disaggregate(): full 7→n_wl disaggregation pipeline.
 *
 * \see include/spectral_brdf.h for the public API.
 */

#include "../include/spectral_brdf.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── MODIS MCD43 band centres [µm] ────────────────────────────────────────── */

const float MODIS_WL_UM[7] = {
    0.469f,   /* B3 blue  */
    0.555f,   /* B4 green */
    0.645f,   /* B1 red   */
    0.858f,   /* B2 NIR   */
    1.240f,   /* B5 SWIR  */
    1.640f,   /* B6 SWIR  */
    2.130f    /* B7 SWIR  */
};

/* ── Piecewise-linear interpolation ──────────────────────────────────────── */

/**
 * Linear interpolation between anchor points; clamp-extrapolates outside range.
 *
 * @param x_anchor  Sorted anchor x-values (length n_anchor).
 * @param y_anchor  Anchor y-values (length n_anchor).
 * @param n_anchor  Number of anchor points.
 * @param x_out     Query x-values (length n_out).
 * @param n_out     Number of query points.
 * @param y_out     Output buffer (caller-allocated, length n_out).
 */
static void spectral_interp_linear(const float *x_anchor, const float *y_anchor,
                                    int n_anchor,
                                    const float *x_out, int n_out,
                                    float *y_out)
{
    for (int j = 0; j < n_out; j++) {
        float x = x_out[j];

        /* Clamp-extrapolate below / above */
        if (x <= x_anchor[0]) {
            y_out[j] = y_anchor[0];
            continue;
        }
        if (x >= x_anchor[n_anchor - 1]) {
            y_out[j] = y_anchor[n_anchor - 1];
            continue;
        }

        /* Binary search for bracket */
        int lo = 0, hi = n_anchor - 2;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (x_anchor[mid + 1] <= x) lo = mid + 1;
            else                         hi = mid;
        }

        float dx = x_anchor[lo + 1] - x_anchor[lo];
        float t  = (dx > 1e-12f) ? (x - x_anchor[lo]) / dx : 0.0f;
        y_out[j] = y_anchor[lo] * (1.0f - t) + y_anchor[lo + 1] * t;
    }
}

/* ── Tikhonov second-difference smoother ─────────────────────────────────── */

/*
 * Solves (I + alpha^2 * D2^T D2) x = f  in-place.
 *
 * D2^T D2 is a symmetric positive semi-definite 5-diagonal matrix with:
 *   main diagonal:  [1, 5, 6, 6, ..., 6, 5, 1]  (boundary corners)
 *   ±1 sub-diag:    [-2, -4, -4, ..., -4, -2]
 *   ±2 sub-diag:    [1, 1, ..., 1]
 *
 * After adding alpha^2 * D2^T D2, the system is symmetric positive-definite
 * (SPD) and banded with bandwidth 2.  We factor A = L L^T where L is lower
 * banded (3 diagonals stored per row) and solve via forward/back substitution.
 *
 * Storage: L[i][0] = main, L[i][1] = sub-1, L[i][2] = sub-2.
 */
void spectral_smooth_tikhonov(float *f, int n, float alpha)
{
    if (n < 3 || alpha <= 0.0f) return;

    float a2 = alpha * alpha;

    /* ── Assemble band A = I + alpha^2 * D2^T D2 ── *
     * Store lower-triangle band: A_main[i], A_sub1[i], A_sub2[i].
     * Use VLAs — maximum n_wl is ~10000, so stack is fine.            */
    float *A0 = (float *)malloc((size_t)n * sizeof(float));
    float *A1 = (float *)malloc((size_t)n * sizeof(float));
    float *A2 = (float *)malloc((size_t)n * sizeof(float));
    if (!A0 || !A1 || !A2) {
        free(A0); free(A1); free(A2);
        return; /* OOM: return unchanged */
    }

    /* Main diagonal of D2^T D2 */
    static const float d2_main_edge0 = 1.0f;
    static const float d2_main_edge1 = 5.0f;
    static const float d2_main_inner = 6.0f;

    for (int i = 0; i < n; i++) {
        float d2_main = (i == 0 || i == n - 1) ? d2_main_edge0
                      : (i == 1 || i == n - 2) ? d2_main_edge1
                      :                           d2_main_inner;
        A0[i] = 1.0f + a2 * d2_main;
    }

    /* ±1 sub-diagonal of D2^T D2 */
    for (int i = 0; i < n - 1; i++) {
        float d2_sub1 = (i == 0 || i == n - 2) ? -2.0f : -4.0f;
        A1[i] = a2 * d2_sub1;   /* A1[i] = A[i+1,i] */
    }
    A1[n - 1] = 0.0f;

    /* ±2 sub-diagonal of D2^T D2 = 1 everywhere */
    for (int i = 0; i < n - 2; i++)
        A2[i] = a2 * 1.0f;
    A2[n - 2] = A2[n - 1] = 0.0f;

    /* ── Cholesky band factorization ── *
     * Overwrites A0,A1,A2 with lower-banded Cholesky factor L. */
    for (int i = 0; i < n; i++) {
        /* Subtract contributions from previous columns */
        if (i >= 1) A0[i] -= A1[i-1] * A1[i-1];
        if (i >= 2) A0[i] -= A2[i-2] * A2[i-2];

        if (A0[i] <= 0.0f) A0[i] = 1e-12f;  /* guard (shouldn't happen for SPD) */
        float inv_d = 1.0f / sqrtf(A0[i]);
        A0[i] = sqrtf(A0[i]);

        if (i + 1 < n) {
            A1[i] -= (i >= 1) ? A2[i-1] * A1[i-1] : 0.0f;
            A1[i] *= inv_d;
        }
        if (i + 2 < n) {
            A2[i] *= inv_d;
        }
    }

    /* ── Forward substitution: L y = f ── */
    for (int i = 0; i < n; i++) {
        if (i >= 1) f[i] -= A1[i-1] * f[i-1];
        if (i >= 2) f[i] -= A2[i-2] * f[i-2];
        f[i] /= A0[i];
    }

    /* ── Back substitution: L^T x = y ── */
    for (int i = n - 1; i >= 0; i--) {
        if (i + 1 < n) f[i] -= A1[i] * f[i+1];
        if (i + 2 < n) f[i] -= A2[i] * f[i+2];
        f[i] /= A0[i];
    }

    free(A0); free(A1); free(A2);
}

/* ── MCD43 7-band disaggregation ─────────────────────────────────────────── */

void mcd43_disaggregate(const float *fiso_7, const float *fvol_7, const float *fgeo_7,
                         const float *wl_target, int n_wl, float alpha,
                         float *fiso_wl, float *fvol_wl, float *fgeo_wl)
{
    /* Interpolate each kernel array from 7 MODIS anchors to n_wl targets */
    spectral_interp_linear(MODIS_WL_UM, fiso_7, 7, wl_target, n_wl, fiso_wl);
    spectral_interp_linear(MODIS_WL_UM, fvol_7, 7, wl_target, n_wl, fvol_wl);
    spectral_interp_linear(MODIS_WL_UM, fgeo_7, 7, wl_target, n_wl, fgeo_wl);

    /* Optional Tikhonov smoothing — applied independently to each kernel */
    if (alpha > 0.0f) {
        spectral_smooth_tikhonov(fiso_wl, n_wl, alpha);
        spectral_smooth_tikhonov(fvol_wl, n_wl, alpha);
        spectral_smooth_tikhonov(fgeo_wl, n_wl, alpha);
    }
}
