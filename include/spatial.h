/* Spatial filtering utilities for atmospheric parameter maps.
 *
 * Used for:
 *   - Gaussian smoothing of per-pixel AOD/H2O raster maps (#1 superpixel-like)
 *   - Box-filter environmental reflectance for adjacency correction (#2)
 *
 * All functions are NaN-safe: NaN pixels are excluded from neighbourhood
 * averages and NaN output pixels are restored where all neighbours were NaN.
 * Boundary pixels use edge-replication (nearest) padding.
 */
#pragma once

/* Separable Gaussian filter applied in-place to a 2D float array [nrows×ncols].
 * sigma: standard deviation in pixels (0 or negative → no-op).
 * Kernel half-width = ceil(3·sigma); edge-replication padding.
 * OpenMP-parallelised over rows. */
void spatial_gaussian_filter(float *data, int nrows, int ncols, float sigma);

/* Separable box (uniform mean) filter.
 * 'data' is the input; 'out' receives the result (must NOT alias 'data').
 * filter_half: half-width in pixels; total window = (2·half+1)².
 * OpenMP-parallelised over rows. */
void spatial_box_filter(const float *data, float *out,
                        int nrows, int ncols, int filter_half);
