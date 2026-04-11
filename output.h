#pragma once
/*
 * output.h
 * ─────────────────────────────────────────────────────────────────────────────
 * Funzioni di analisi e output della dose 3D.
 * Produce PDD, profilo laterale, heatmap e statistiche.
 *
 * Autore: Angelica Porco - Matricola 264034
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cstdio>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "physics.h"

// ─────────────────────────────────────────────────────────────────────────────
// PDD  —  Percentage Depth Dose lungo l'asse Z
// Media su una finestra ±avg_half voxel attorno all'asse centrale (X=NY/2)
// ─────────────────────────────────────────────────────────────────────────────
inline void compute_pdd(const double *dose,
                         double *pdd,       // output: NZ valori in [0,100]
                         double *depths,    // output: NZ profondità [cm]
                         int avg_half = 8)  // ±8 voxel = ±2.4cm ≈ campo 5x5cm
{
    int cx = NX / 2;
    int cy = NY / 2;

    double max_dose = 0.0;
    for (int iz = 0; iz < NZ; iz++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int ix = cx - avg_half; ix <= cx + avg_half; ix++)
        for (int iy = cy - avg_half; iy <= cy + avg_half; iy++) {
            if (ix >= 0 && ix < NX && iy >= 0 && iy < NY) {
                sum += dose[phantom_idx(ix, iy, iz)];
                cnt++;
            }
        }
        pdd[iz]    = (cnt > 0) ? sum / cnt : 0.0;
        depths[iz] = (iz + 0.5) * VOXEL_CM;
        if (pdd[iz] > max_dose) max_dose = pdd[iz];
    }

    // Normalizza al massimo → percentuale
    if (max_dose > 0.0)
        for (int iz = 0; iz < NZ; iz++)
            pdd[iz] = pdd[iz] / max_dose * 100.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// PROFILO LATERALE a profondità fissa (lungo X, centrato su Y)
// ─────────────────────────────────────────────────────────────────────────────
inline void compute_lateral_profile(const double *dose,
                                     double *profile,  // output: NX valori in [0,100]
                                     double *positions, // output: NX posizioni [cm]
                                     double depth_cm = 10.0,
                                     int avg_half = 2)
{
    int iz = std::min((int)(depth_cm / VOXEL_CM), NZ - 1);
    int cy = NY / 2;

    double max_val = 0.0;
    for (int ix = 0; ix < NX; ix++) {
        double sum = 0.0;
        int    cnt = 0;
        for (int iy = cy - avg_half; iy <= cy + avg_half; iy++) {
            if (iy >= 0 && iy < NY) {
                sum += dose[phantom_idx(ix, iy, iz)];
                cnt++;
            }
        }
        profile[ix]   = (cnt > 0) ? sum / cnt : 0.0;
        positions[ix] = (ix + 0.5) * VOXEL_CM - PHANTOM_CM / 2.0;
        if (profile[ix] > max_val) max_val = profile[ix];
    }

    if (max_val > 0.0)
        for (int ix = 0; ix < NX; ix++)
            profile[ix] = profile[ix] / max_val * 100.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// SALVA PDD SU CSV
// ─────────────────────────────────────────────────────────────────────────────
inline void save_pdd_csv(const double *depths, const double *pdd,
                          const char *filename) {
    std::ofstream f(filename);
    f << "depth_cm,dose_percent\n";
    for (int iz = 0; iz < NZ; iz++)
        f << depths[iz] << "," << pdd[iz] << "\n";
    f.close();
    printf("  Salvato: %s\n", filename);
}

// ─────────────────────────────────────────────────────────────────────────────
// SALVA PROFILO LATERALE SU CSV
// ─────────────────────────────────────────────────────────────────────────────
inline void save_profile_csv(const double *positions, const double *profile,
                              const char *filename) {
    std::ofstream f(filename);
    f << "position_cm,dose_percent\n";
    for (int ix = 0; ix < NX; ix++)
        f << positions[ix] << "," << profile[ix] << "\n";
    f.close();
    printf("  Salvato: %s\n", filename);
}

// ─────────────────────────────────────────────────────────────────────────────
// SALVA SLICE 2D CENTRALE SU CSV (per heatmap)
// ─────────────────────────────────────────────────────────────────────────────
inline void save_dose_slice_csv(const double *dose, const char *filename) {
    std::ofstream f(filename);
    int iy = NY / 2;
    for (int iz = 0; iz < NZ; iz++) {
        for (int ix = 0; ix < NX; ix++) {
            f << dose[phantom_idx(ix, iy, iz)];
            if (ix < NX - 1) f << ",";
        }
        f << "\n";
    }
    f.close();
    printf("  Salvato: %s\n", filename);
}

// ─────────────────────────────────────────────────────────────────────────────
// SALVA DOSE 3D COMPLETA (formato binario float64)
// ─────────────────────────────────────────────────────────────────────────────
inline void save_dose_binary(const double *dose, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) { printf("ERRORE: impossibile aprire %s\n", filename); return; }
    fwrite(dose, sizeof(double), NX * NY * NZ, f);
    fclose(f);
    printf("  Salvato: %s  (%d float64)\n", filename, NX * NY * NZ);
}

// ─────────────────────────────────────────────────────────────────────────────
// STAMPA TABELLA PDD AI PUNTI DI RIFERIMENTO CLINICI
// ─────────────────────────────────────────────────────────────────────────────
inline void print_pdd_table(const double *depths, const double *pdd,
                             const char *label) {
    printf("\n  PDD — %s\n", label);
    printf("  %-20s  %10s  %s\n", "Profondità [cm]", "Dose [%]", "Riferimento");
    printf("  %s\n", "─────────────────────────────────────────");

    double refs[]      = {1.5, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0};
    const char *notes[] = {"build-up", "~d_max 6MV", "", "D10", "", "D20", ""};

    for (int r = 0; r < 7; r++) {
        int k = (int)(refs[r] / VOXEL_CM);
        if (k >= 0 && k < NZ)
            printf("  %-20.1f  %10.2f  %s\n", depths[k], pdd[k], notes[r]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// STATISTICHE GENERALI SULLA DOSE
// ─────────────────────────────────────────────────────────────────────────────
inline void print_dose_stats(const double *dose, long long N_simulated,
                              double elapsed_sec) {
    double max_d = 0.0, total_e = 0.0;
    int    nonzero = 0;

    for (int i = 0; i < NX * NY * NZ; i++) {
        if (dose[i] > 0.0) {
            nonzero++;
            total_e += dose[i];
            if (dose[i] > max_d) max_d = dose[i];
        }
    }

    printf("\n  ── Statistiche simulazione ──────────────────\n");
    printf("  Particelle simulate : %lld\n",  N_simulated);
    printf("  Tempo totale        : %.2f s\n", elapsed_sec);
    printf("  Throughput          : %.3f MP/s\n",
           N_simulated / elapsed_sec / 1.0e6);
    printf("  Tempo/particella    : %.1f μs\n",
           elapsed_sec / N_simulated * 1.0e6);
    printf("  Voxel con dose>0    : %d / %d (%.1f%%)\n",
           nonzero, NX*NY*NZ, 100.0*nonzero/(NX*NY*NZ));
    printf("  Energia totale dep. : %.4e MeV\n", total_e);
    printf("  Energia/particella  : %.4e MeV\n",
           N_simulated > 0 ? total_e / N_simulated : 0.0);
    printf("  Dose massima (u.a.) : %.6e\n", max_d);
}
