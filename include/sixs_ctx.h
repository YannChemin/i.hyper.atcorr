/* 6SV2.1 → C port: internal context replacing COMMON blocks.
 * Each LUT computation thread gets its own SixsCtx instance. */
#pragma once
#include <stdbool.h>
#include <stdint.h>

/* Dimensions matching paramdef.inc */
#define MU_P      25      /* streams per hemisphere (mu_p) */
#define NT_P      30      /* max atmospheric layers (nt_p) */
#define NP_P      49      /* phi grid points (np_p) */
#define NQ_P      83      /* default Gauss quadrature points (nqdef_p) */
#define NQ_MAX    1001    /* max quadrature points (nqmax_p) */
#define NWL_DISC  20      /* number of discrete reference wavelengths */

/* Standard atmosphere: 34 layers (ODRAYL / ABSTRA use this) */
#define NATM      34

/* /sixs_atm/ */
typedef struct {
    float z[NATM];   /* altitude km */
    float p[NATM];   /* pressure hPa */
    float t[NATM];   /* temperature K */
    float wh[NATM];  /* water vapour g/m³ */
    float wo[NATM];  /* ozone g/m³ */
} SixsAtm;

/* /sixs_del/ depolarization */
typedef struct {
    float delta;   /* depolarization factor */
    float sigma;   /* (unused in scalar mode) */
} SixsDel;

/* /sixs_aer/ aerosol optical properties at 20 reference wavelengths */
typedef struct {
    float ext[NWL_DISC];    /* extinction coefficient (normalized) */
    float ome[NWL_DISC];    /* single scattering albedo */
    float gasym[NWL_DISC];  /* asymmetry parameter */
    float phase[NWL_DISC];  /* phase function at scattering angle */
} SixsAer;

/* /sixs_polar/ Legendre coefficients from TRUNCA */
typedef struct {
    float pha[NQ_P];              /* phase function at Gauss points */
    float alphal[NQ_P + 1];       /* Legendre coefficients (polarized) */
    float betal[NQ_P + 1];        /* Legendre coefficients (scalar) */
    float gammal[NQ_P + 1];
    float zetal[NQ_P + 1];
} SixsPolar;

/* /sixs_disc/ DISCOM output: pre-computed at 20 reference wavelengths */
typedef struct {
    float roatm[3][NWL_DISC];    /* atmospheric reflectance [ray,mix,aer] */
    float dtdir[3][NWL_DISC];    /* downward direct transmittance */
    float dtdif[3][NWL_DISC];    /* downward diffuse transmittance */
    float utdir[3][NWL_DISC];    /* upward direct transmittance */
    float utdif[3][NWL_DISC];    /* upward diffuse transmittance */
    float sphal[3][NWL_DISC];    /* spherical albedo */
    float wldis[NWL_DISC];       /* reference wavelengths (µm) */
    float trayl[NWL_DISC];       /* Rayleigh optical depth */
    float traypl[NWL_DISC];      /* Rayleigh OD above sensor level */
} SixsDisc;

/* /num_quad/ */
typedef struct {
    int nquad;   /* current number of Gauss quadrature points */
} SixsQuad;

/* /multorder/ */
typedef struct {
    int igmax;   /* max scattering orders (default 20) */
} SixsMultiOrder;

/* Error state */
typedef struct {
    int  iwr;    /* output unit (unused in C, always stdout) */
    bool ier;    /* error flag */
} SixsErr;

/* Master context struct — one per thread */
typedef struct {
    SixsAtm        atm;
    SixsDel        del;
    SixsAer        aer;
    SixsPolar      polar;
    SixsDisc       disc;
    SixsQuad       quad;
    SixsMultiOrder multi;
    SixsErr        err;
    /* Gauss quadrature points/weights for rm/gb arrays (size NQ_P) */
    float rm[2 * MU_P + 1];   /* cosines, index [-MU_P .. MU_P] offset by MU_P */
    float gb[2 * MU_P + 1];   /* weights */
} SixsCtx;

/* Convenience offset macros for negative-indexed arrays */
#define RM(ctx, i)  ((ctx)->rm[(i) + MU_P])
#define GB(ctx, i)  ((ctx)->gb[(i) + MU_P])
