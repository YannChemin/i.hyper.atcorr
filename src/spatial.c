/* Spatial filtering: separable Gaussian and box filters.
 * See include/spatial.h for API documentation. */

#include "spatial.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Box filter ─────────────────────────────────────────────────────────────── */

void spatial_box_filter(const float *data, float *out,
                        int nrows, int ncols, int filter_half)
{
    if (filter_half < 1) {
        memcpy(out, data, (size_t)nrows * ncols * sizeof(float));
        return;
    }

    float *tmp = malloc((size_t)nrows * ncols * sizeof(float));
    if (!tmp) return;

    /* ── Horizontal pass: data → tmp ── */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            double sum = 0.0;
            int    n   = 0;
            for (int k = -filter_half; k <= filter_half; k++) {
                int cc = c + k;
                if (cc < 0)      cc = 0;
                if (cc >= ncols) cc = ncols - 1;
                float v = data[r * ncols + cc];
                if (!isnan(v)) { sum += v; n++; }
            }
            tmp[r * ncols + c] = (n > 0)
                ? (float)(sum / n)
                : data[r * ncols + c];
        }
    }

    /* ── Vertical pass: tmp → out ── */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            double sum = 0.0;
            int    n   = 0;
            for (int k = -filter_half; k <= filter_half; k++) {
                int rr = r + k;
                if (rr < 0)      rr = 0;
                if (rr >= nrows) rr = nrows - 1;
                float v = tmp[rr * ncols + c];
                if (!isnan(v)) { sum += v; n++; }
            }
            out[r * ncols + c] = (n > 0)
                ? (float)(sum / n)
                : tmp[r * ncols + c];
        }
    }

    free(tmp);
}

/* ── Gaussian filter ─────────────────────────────────────────────────────────── */

void spatial_gaussian_filter(float *data, int nrows, int ncols, float sigma)
{
    if (sigma <= 0.0f) return;

    int radius = (int)(3.0f * sigma + 0.5f);
    if (radius < 1) radius = 1;
    int ksize = 2 * radius + 1;

    float *kernel = malloc(ksize * sizeof(float));
    if (!kernel) return;

    /* Build normalised Gaussian kernel */
    float ksum = 0.0f;
    for (int k = -radius; k <= radius; k++) {
        float v = expf(-0.5f * (float)(k * k) / (sigma * sigma));
        kernel[k + radius] = v;
        ksum += v;
    }
    for (int k = 0; k < ksize; k++) kernel[k] /= ksum;

    float *tmp = malloc((size_t)nrows * ncols * sizeof(float));
    if (!tmp) { free(kernel); return; }

    /* ── Horizontal pass: data → tmp ── */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            float val = 0.0f, w = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                int cc = c + k;
                if (cc < 0)      cc = 0;
                if (cc >= ncols) cc = ncols - 1;
                float v = data[r * ncols + cc];
                if (!isnan(v)) {
                    float kw = kernel[k + radius];
                    val += kw * v;
                    w   += kw;
                }
            }
            tmp[r * ncols + c] = (w > 1e-6f) ? val / w : data[r * ncols + c];
        }
    }

    /* ── Vertical pass: tmp → data (in-place result) ── */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            float val = 0.0f, w = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                int rr = r + k;
                if (rr < 0)      rr = 0;
                if (rr >= nrows) rr = nrows - 1;
                float v = tmp[rr * ncols + c];
                if (!isnan(v)) {
                    float kw = kernel[k + radius];
                    val += kw * v;
                    w   += kw;
                }
            }
            data[r * ncols + c] = (w > 1e-6f) ? val / w : tmp[r * ncols + c];
        }
    }

    free(tmp);
    free(kernel);
}
