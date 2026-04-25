
#pragma once

#include <cstdio>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "physics.cuh"

// PDD
inline void compute_pdd(const double *dose, double *pdd, double *profondita, int semi = 8) {
    int cx = NX / 2;
    int cy = NY / 2;

    double max_dose = 0.0;
    for (int iz = 0; iz < NZ; iz++) {
        double val  = 0.0;
        int    cnt  = 0;
        for (int ix = cx - semi; ix <= cx + semi; ix++)
        for (int iy = cy - semi; iy <= cy + semi; iy++) {
            if (ix >= 0 && ix < NX && iy >= 0 && iy < NY) {
                val += dose[phantom_idx(ix, iy, iz)];
                cnt++;
            }
        }
        pdd[iz]      = (cnt > 0) ? val / cnt : 0.0;
        profondita[iz] = (iz + 0.5) * VOXEL_CM;
        if (pdd[iz] > max_dose) max_dose = pdd[iz];
    }
    if (max_dose > 0.0)
        for (int iz = 0; iz < NZ; iz++)
            pdd[iz] = pdd[iz] / max_dose * 100.0;
}

// PROFILO LATERALE
inline void compute_lateral_profile(const double *dose, double *profilo, double *coord,
                                     double profondita_scelta = 10.0, int semi = 2) {
    int iz  = std::min((int)(profondita_scelta / VOXEL_CM), NZ - 1);
    int cy  = NY / 2;
    double dmax = 0.0;

    for (int ix = 0; ix < NX; ix++) {
        double s = 0.0; int c = 0;
        for (int iy = cy - semi; iy <= cy + semi; iy++) {
            if (iy >= 0 && iy < NY) { s += dose[phantom_idx(ix, iy, iz)]; c++; }
        }
        profilo[ix] = (c > 0) ? s / c : 0.0;
        coord[ix]   = (ix + 0.5) * VOXEL_CM - PHANTOM_CM / 2.0;
        if (profilo[ix] > dmax) dmax = profilo[ix];
    }
    if (dmax > 0.0)
        for (int ix = 0; ix < NX; ix++)
            profilo[ix] = profilo[ix] / dmax * 100.0;
}

inline void save_pdd_csv(const double *depth, const double *pdd, const char *fn) {
    std::ofstream f(fn);
    f << "depth_cm,dose_percent\n";
    for (int iz = 0; iz < NZ; iz++) f << depth[iz] << "," << pdd[iz] << "\n";
    f.close();
    printf("Salvato: %s\n", fn);
}

inline void save_profile_csv(const double *coord, const double *profilo, const char *fn) {
    std::ofstream f(fn);
    f << "position_cm,dose_percent\n";
    for (int ix = 0; ix < NX; ix++) f << coord[ix] << "," << profilo[ix] << "\n";
    f.close();
    printf("  Salvato: %s\n", fn);
}

inline void save_dose_slice_csv(const double *dose, const char *fn) {
    std::ofstream f(fn);
    int iy = NY / 2;
    for (int iz = 0; iz < NZ; iz++) {
        for (int ix = 0; ix < NX; ix++) {
            f << dose[phantom_idx(ix, iy, iz)];
            if (ix < NX - 1) f << ",";
        }
        f << "\n";
    }
    f.close();
    printf("  Salvato: %s\n", fn);
}

inline void save_dose_binary(const double *dose, const char *fn) {
    FILE *f = fopen(fn, "wb");
    if (!f) { printf("ERRORE: impossibile aprire %s\n", fn); return; }
    fwrite(dose, sizeof(double), NX * NY * NZ, f);
    fclose(f);
    printf("Salvato: %s  (%d float64)\n", fn, NX * NY * NZ);
}

inline void print_pdd_table(const double *profondita, const double *pdd, const char *label) {
    printf("\n  PDD — %s\n", label);
    printf("  %-20s  %10s  %s\n", "Profondità [cm]", "Dose [%]", "Riferimento");
    printf("  %s\n", "─────────────────────────────────────────");
    double refs[]       = {1.5, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0};
    const char *notes[] = {"build-up", "d_max 6MV", "", "D10", "", "D20", ""};
    for (int r = 0; r < 7; r++) {
        int k = (int)(refs[r] / VOXEL_CM);
        if (k >= 0 && k < NZ)
            printf("  %-20.1f  %10.2f  %s\n", profondita[k], pdd[k], notes[r]);
    }
}

inline void print_dose_stats(const double *dose, long long n_part, double t_sec) {
    double dmax = 0.0, etot = 0.0;
    int nhit = 0;
    for (int i = 0; i < NX * NY * NZ; i++) {
        if (dose[i] > 0.0) { nhit++; etot += dose[i]; if (dose[i] > dmax) dmax = dose[i]; }
    }
    printf("\n Statistiche simulazione: \n");
    printf("  Particelle simulate : %lld\n",  n_part);
    printf("  Tempo totale        : %.2f s\n", t_sec);
    printf("  Throughput          : %.3f MP/s\n", n_part / t_sec / 1.0e6);
    printf("  Tempo/particella    : %.1f μs\n", t_sec / n_part * 1.0e6);
    printf("  Voxel con dose>0    : %d / %d (%.1f%%)\n", nhit, NX*NY*NZ, 100.0*nhit/(NX*NY*NZ));
    printf("  Energia totale dep. : %.4e MeV\n", etot);
    printf("  Energia/particella  : %.4e MeV\n", n_part > 0 ? etot / n_part : 0.0);
    printf("  Dose massima (u.a.) : %.6e\n", dmax);
}
