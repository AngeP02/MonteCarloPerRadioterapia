#pragma once

#include <cmath>
#include <cassert>

// COSTANTI FISICHE
static const double ME_C2    = 0.51099895;
static const double PI       = 3.14159265358979323846;
static const double ECUT     = 0.010;
static const double PCUT     = 0.100;

// GEOMETRIA PHANTOM
static const int    NX = 100, NY = 100, NZ = 100;
static const double VOXEL_CM   = 0.30;
static const double PHANTOM_CM = NX * VOXEL_CM;

// INDICI MATERIALI
#define MAT_WATER 0
#define MAT_BONE  1
#define MAT_LUNG  2
#define MAT_AIR   3
#define N_MAT     4

// DENSITÀ [g/cm^3]
__constant__ double d_DENSITY[N_MAT] = { 1.000, 1.850, 0.260, 0.001205 };

// GRIGLIA ENERGETICA [MeV] (28 punti, da 0.01 a 20 MeV)
static const int N_ENERGY = 28;
__constant__ double d_ENERGY_GRID[N_ENERGY] = {
    0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100,
    0.150, 0.200, 0.300, 0.400, 0.500, 0.600, 0.800, 1.000, 1.250,
    1.500, 2.000, 3.000, 4.000, 5.000, 6.000, 8.000, 10.000,
    15.000, 20.000
};

__constant__ double d_MU_TOTAL[N_MAT][N_ENERGY] = {
    { 5.329, 1.673, 0.8096, 0.3756, 0.2683, 0.2269, 0.2059, 0.1837, 0.1707,
      0.1505, 0.1370, 0.1186, 0.1061, 0.09687, 0.09007, 0.07865, 0.07072, 0.06323,
      0.05754, 0.04942, 0.03969, 0.03403, 0.03031, 0.02770, 0.02429, 0.02219,
      0.01941, 0.01813 },
    { 19.89, 7.131, 3.085, 1.012, 0.5475, 0.3941, 0.3178, 0.2595, 0.2368,
      0.1958, 0.1698, 0.1393, 0.1222, 0.1107, 0.1018, 0.08795, 0.07838, 0.06945,
      0.06283, 0.05351, 0.04257, 0.03624, 0.03209, 0.02913, 0.02536, 0.02296,
      0.01978, 0.01832 },
    { 5.208, 1.638, 0.7933, 0.3681, 0.2629, 0.2224, 0.2018, 0.1800, 0.1673,
      0.1475, 0.1342, 0.1162, 0.1040, 0.09493, 0.08827, 0.07708, 0.06930, 0.06196,
      0.05639, 0.04843, 0.03889, 0.03335, 0.02970, 0.02715, 0.02380, 0.02175,
      0.01902, 0.01776 },
    { 36.01, 12.28, 5.279, 1.625, 0.6918, 0.3954, 0.2885, 0.2085, 0.1875,
      0.1504, 0.1340, 0.1123, 0.09921, 0.09076, 0.08414, 0.07285, 0.06529, 0.05817,
      0.05298, 0.04534, 0.03597, 0.03054, 0.02699, 0.02453, 0.02131, 0.01931,
      0.01673, 0.01551 }
};

__constant__ double d_MU_PE[N_MAT][N_ENERGY] = {
    { 4.944, 1.374, 0.5195, 0.1036, 0.02407, 0.005800, 0.001334, 5.510e-5, 3.998e-5,
      2.799e-6, 2.200e-7, 1.400e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 19.35, 6.833, 2.818, 0.7469, 0.2837, 0.1152, 0.04660, 0.008680, 0.001900,
      1.800e-4, 2.000e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 4.845, 1.346, 0.5091, 0.1015, 0.02359, 0.005684, 0.001307, 5.400e-5, 3.918e-5,
      2.743e-6, 2.156e-7, 1.372e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 35.52, 11.99, 5.012, 1.379, 0.4529, 0.1581, 0.05757, 0.008251, 0.001581,
      8.208e-5, 7.636e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
};

__constant__ double d_MU_COMPTON[N_MAT][N_ENERGY] = {
    { 0.3854, 0.2988, 0.2672, 0.2651, 0.2595, 0.2476, 0.2329, 0.1984, 0.1661,
      0.1505, 0.1370, 0.1186, 0.1061, 0.09687, 0.09007, 0.07865, 0.07072, 0.06323,
      0.05754, 0.04942, 0.03969, 0.03403, 0.03031, 0.02770, 0.02429, 0.02219,
      0.01878, 0.01719 },
    { 0.4869, 0.2684, 0.2503, 0.2465, 0.2429, 0.2310, 0.2172, 0.1848, 0.1548,
      0.1400, 0.1275, 0.1103, 0.09870, 0.09010, 0.08377, 0.07313, 0.06575, 0.05862,
      0.05338, 0.04579, 0.03667, 0.03133, 0.02784, 0.02539, 0.02217, 0.02016,
      0.01702, 0.01552 },
    { 0.3777, 0.2928, 0.2619, 0.2598, 0.2543, 0.2426, 0.2282, 0.1944, 0.1628,
      0.1475, 0.1342, 0.1162, 0.1040, 0.09493, 0.08827, 0.07708, 0.06930, 0.06196,
      0.05639, 0.04843, 0.03889, 0.03335, 0.02970, 0.02715, 0.02380, 0.02175,
      0.01840, 0.01684 },
    { 0.3779, 0.2933, 0.2624, 0.2602, 0.2547, 0.2430, 0.2285, 0.1946, 0.1630,
      0.1477, 0.1344, 0.1163, 0.1041, 0.09516, 0.08844, 0.07723, 0.06942, 0.06207,
      0.05649, 0.04852, 0.03894, 0.03339, 0.02973, 0.02718, 0.02382, 0.02177,
      0.01843, 0.01686 }
};

__constant__ double d_MU_PAIR[N_MAT][N_ENERGY] = {
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.000630, 0.000940 },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.002760, 0.002800 },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.000617, 0.000921 },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.000589, 0.000879 }
};

// INTERPOLAZIONE LINEARE SU GRIGLIA ENERGETICA (device)
__device__ inline double interp_mu_dev(double energia_mev, const double *tabella) {
    if (energia_mev <= d_ENERGY_GRID[0])       return tabella[0];
    if (energia_mev >= d_ENERGY_GRID[N_ENERGY-1]) return tabella[N_ENERGY-1];

    int lo = 0, hi = N_ENERGY - 1;
    while (hi - lo > 1) {
        int m = (lo + hi) / 2;
        if (d_ENERGY_GRID[m] <= energia_mev) lo = m; else hi = m;
    }
    double t = (energia_mev - d_ENERGY_GRID[lo]) / (d_ENERGY_GRID[hi] - d_ENERGY_GRID[lo]);
    return tabella[lo] * (1.0 - t) + tabella[hi] * t;
}

__device__ inline double get_mu_total_dev(double e, int mat) {
    return interp_mu_dev(e, d_MU_TOTAL[mat]) * d_DENSITY[mat];
}
__device__ inline double get_mu_pe_dev(double e, int mat) {
    return interp_mu_dev(e, d_MU_PE[mat]) * d_DENSITY[mat];
}
__device__ inline double get_mu_compton_dev(double e, int mat) {
    return interp_mu_dev(e, d_MU_COMPTON[mat]) * d_DENSITY[mat];
}
__device__ inline double get_mu_pair_dev(double e, int mat) {
    return interp_mu_dev(e, d_MU_PAIR[mat]) * d_DENSITY[mat];
}

__device__ inline int select_interaction_dev(double e, int mat, double xi) {
    double mu_tot = get_mu_total_dev(e, mat);
    if (mu_tot <= 0.0) return 1;
    double pfe = get_mu_pe_dev(e, mat)      / mu_tot;
    double pco = get_mu_compton_dev(e, mat) / mu_tot;
    if (xi < pfe)        return 0;
    if (xi < pfe + pco)  return 1;
    return 2;
}

__device__ inline int phantom_idx_dev(int ix, int iy, int iz) {
    return ix + NX * (iy + NY * iz);
}

__device__ inline bool inside_dev(double x, double y, double z) {
    return x >= 0.0 && x < PHANTOM_CM &&
           y >= 0.0 && y < PHANTOM_CM &&
           z >= 0.0 && z < PHANTOM_CM;
}

__device__ inline int vox_dev(double coord) {
    int v = (int)(coord / VOXEL_CM);
    if (v < 0)  v = 0;
    if (v >= NX) v = NX - 1;
    return v;
}

// ── Helper CPU (host) ── usati da phantom.cuh e output.cuh ──────────────────
inline int phantom_idx(int ix, int iy, int iz) {
    return ix + NX * (iy + NY * iz);
}

inline bool inside(double x, double y, double z) {
    return x >= 0.0 && x < PHANTOM_CM &&
           y >= 0.0 && y < PHANTOM_CM &&
           z >= 0.0 && z < PHANTOM_CM;
}

inline int vox(double coord) {
    int v = (int)(coord / VOXEL_CM);
    if (v < 0)  v = 0;
    if (v >= NX) v = NX - 1;
    return v;
}
