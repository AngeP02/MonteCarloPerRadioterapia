
#pragma once

#include <cstdio>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "physics.h"

// PDD
inline void compute_pdd(const double *dose, double *pdd, double *profondita, int semiampiezza_della_media = 8) {
    int cx = NX / 2; // centro del fascio in X
    int cy = NY / 2; // centro del fascio in Y

    double max_dose = 0.0;
    for (int iz = 0; iz < NZ; iz++) {
        double valore_dose = 0.0;
        int    num_voxel_sommati = 0;
        for (int ix = cx - semiampiezza_della_media; ix <= cx + semiampiezza_della_media; ix++){
          for (int iy = cy - semiampiezza_della_media; iy <= cy + semiampiezza_della_media; iy++) {
              if (ix >= 0 && ix < NX && iy >= 0 && iy < NY) {
                  valore_dose += dose[phantom_idx(ix, iy, iz)];
                  num_voxel_sommati++;
              }
          }
        }
        if (num_voxel_sommati > 0) {
            pdd[iz] = valore_dose / num_voxel_sommati;
        } else{
            0.0;
        }
        profondita[iz] = (iz + 0.5) * VOXEL_CM;
        if (pdd[iz] > max_dose){
            max_dose = pdd[iz];
        }
    }

    if (max_dose > 0.0)
        for (int iz = 0; iz < NZ; iz++)
            pdd[iz] = pdd[iz] / max_dose * 100.0;
}

// PROFILO LATERALE a profondità fissa (lungo X, centrato su Y)
inline void compute_lateral_profile(const double *dose, double *profilo_dose_relativa, double *coordinate_cm, double profondita_scelta = 10.0, int semiampiezza_media = 2) {
    int iz = std::min((int)(profondita_scelta / VOXEL_CM), NZ - 1);
    int centro_asse_y = NY / 2;

    double dose_massima_profilo = 0.0;
    for (int ix = 0; ix < NX; ix++) {
        double somma_dose_accumulata = 0.0;
        int conteggio_voxel_validi = 0;
        for (int iy = centro_asse_y - semiampiezza_media; iy <= centro_asse_y + semiampiezza_media; iy++) {
            if (iy >= 0 && iy < NY) {
                somma_dose_accumulata += dose[phantom_idx(ix, iy, iz)];
                conteggio_voxel_validi++;
            }
        }
        if (conteggio_voxel_validi > 0){
            profilo_dose_relativa[ix] = somma_dose_accumulata / conteggio_voxel_validi;
        } else{
            profilo_dose_relativa[ix] = 0.0;
        }
        coordinate_cm[ix] = (ix + 0.5) * VOXEL_CM - PHANTOM_CM / 2.0;
        if (profilo_dose_relativa[ix] > dose_massima_profilo) {
            dose_massima_profilo = profilo_dose_relativa[ix];
        }
    }

    if (dose_massima_profilo > 0.0)
        for (int ix = 0; ix < NX; ix++)
            profilo_dose_relativa[ix] = profilo_dose_relativa[ix] / dose_massima_profilo * 100.0;
}

// SALVA PDD SU CSV
inline void save_pdd_csv(const double *vettore_profondita_cm, const double *pdd, const char *filename) {
    std::ofstream f(filename);
    f << "depth_cm,dose_percent\n";
    for (int iz = 0; iz < NZ; iz++)
        f << vettore_profondita_cm[iz] << "," << pdd[iz] << "\n";
    f.close();
    printf("Salvato: %s\n", filename);
}

// SALVA PROFILO LATERALE SU CSV
inline void save_profile_csv(const double *coordinate_cm, const double *profilo_dose_relativa, const char *filename) {
    std::ofstream f(filename);
    f << "position_cm,dose_percent\n";
    for (int ix = 0; ix < NX; ix++)
        f << coordinate_cm[ix] << "," << profilo_dose_relativa[ix] << "\n";
    f.close();
    printf("  Salvato: %s\n", filename);
}

// SALVA SLICE 2D CENTRALE SU CSV (per heatmap)
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

// SALVA DOSE 3D COMPLETA
inline void save_dose_binary(const double *dose, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) { printf("ERRORE: impossibile aprire %s\n", filename); return; }
    fwrite(dose, sizeof(double), NX * NY * NZ, f);
    fclose(f);
    printf("Salvato: %s  (%d float64)\n", filename, NX * NY * NZ);
}

// STAMPA TABELLA PDD AI PUNTI DI RIFERIMENTO CLINICI
inline void print_pdd_table(const double *profondita, const double *pdd, const char *label) {
    printf("\n  PDD — %s\n", label);
    printf("  %-20s  %10s  %s\n", "Profondità [cm]", "Dose [%]", "Riferimento");
    printf("  %s\n", "─────────────────────────────────────────");

    double refs[]      = {1.5, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0};
    const char *notes[] = {"build-up", "d_max 6MV", "", "D10", "", "D20", ""};

    for (int r = 0; r < 7; r++) {
        int k = (int)(refs[r] / VOXEL_CM);
        if (k >= 0 && k < NZ)
            printf("  %-20.1f  %10.2f  %s\n", profondita[k], pdd[k], notes[r]);
    }
}

// STATISTICHE GENERALI SULLA DOSE
inline void print_dose_stats(const double *dose, long long numero_particelle_totali, double tempo_esecuzione_secondi) {
    double dose_massima_assoluta = 0.0;
    double energia_totale_depositata = 0.0;
    int conteggio_voxel_colpiti = 0;

    for (int i = 0; i < NX * NY * NZ; i++) {
        if (dose[i] > 0.0) {
            conteggio_voxel_colpiti++;
            energia_totale_depositata += dose[i];
            if (dose[i] > dose_massima_assoluta){
                dose_massima_assoluta = dose[i];
            }
        }
    }

    printf("\n Statistiche simulazione: \n");
    printf("  Particelle simulate : %lld\n",  numero_particelle_totali);
    printf("  Tempo totale        : %.2f s\n", tempo_esecuzione_secondi);
    printf("  Throughput          : %.3f MP/s\n", numero_particelle_totali / tempo_esecuzione_secondi / 1.0e6);
    printf("  Tempo/particella    : %.1f μs\n", tempo_esecuzione_secondi / numero_particelle_totali * 1.0e6);
    printf("  Voxel con dose>0    : %d / %d (%.1f%%)\n", conteggio_voxel_colpiti, NX*NY*NZ, 100.0*conteggio_voxel_colpiti/(NX*NY*NZ));
    printf("  Energia totale dep. : %.4e MeV\n", energia_totale_depositata);
    printf("  Energia/particella  : %.4e MeV\n", numero_particelle_totali > 0 ? energia_totale_depositata / numero_particelle_totali : 0.0);
    printf("  Dose massima (u.a.) : %.6e\n", dose_massima_assoluta);
}
