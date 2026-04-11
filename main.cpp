/*
 * main.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * Monte Carlo per Radioterapia — Versione CPU Sequenziale
 *
 * Implementa il trasporto di fotoni in un phantom voxelizzato con:
 *   - Spettro 6MV (Sheikh-Bagheri & Rogers 2002)
 *   - Legge di Beer-Lambert + voxel traversal (Amanatides & Woo 1987)
 *   - Effetto fotoelettrico, Compton (metodo di Kahn), produzione coppie
 *   - Sezioni d'urto da NIST XCOM (Hubbell & Seltzer 1996)
 *   - Approssimazione KERMA ≈ dose (deposito locale elettroni)
 *   - PRNG: xoshiro256** (Blackman & Vigna 2018)
 *
 * Uso su Colab:
 *   !g++ -O2 -std=c++17 -o mc_rt_cpu main.cpp -lm
 *   !./mc_rt_cpu [N_fotoni] [phantom: 0=acqua 1=etero] [seed]
 *
 * Autore: Angelica Porco - Matricola 264034
 * Corso:  High Performance Computing 2025/2026
 * Prof.   Domenico Talia  |  Ing. Riccardo Cantini
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>
#include <vector>

#include "physics.h"
#include "compton.h"
#include "random.h"
#include "phantom.h"
#include "output.h"

// ─────────────────────────────────────────────────────────────────────────────
// STRUTTURA: stato di una particella sullo stack
// ─────────────────────────────────────────────────────────────────────────────
struct Particle {
    double x, y, z;    // posizione [cm]
    double ux, uy, uz; // versore direzione (normalizzato)
    double energy;      // energia [MeV]
};

// ─────────────────────────────────────────────────────────────────────────────
// CAMPIONAMENTO SORGENTE
// Fascio parallelo lungo +z, campo 10×10cm² uniforme centrato sul phantom.
// La sorgente è posizionata appena dentro la superficie superiore del phantom.
// ─────────────────────────────────────────────────────────────────────────────
inline Particle sample_source(const Spectrum &spec, Xoshiro256 &rng) {
    static const double FIELD_HALF = 5.0;   // ±5 cm → campo 10×10 cm²
    double cx = PHANTOM_CM / 2.0;
    double cy = PHANTOM_CM / 2.0;

    Particle p;
    p.x      = cx + (rng() * 2.0 - 1.0) * FIELD_HALF;
    p.y      = cy + (rng() * 2.0 - 1.0) * FIELD_HALF;
    p.z      = 1.0e-7;   // appena dentro il phantom
    p.ux     = 0.0;
    p.uy     = 0.0;
    p.uz     = 1.0;      // fascio lungo +z
    p.energy = spec.sample(rng);
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// DISTANZA AL PROSSIMO CONFINE DI VOXEL
// Implementazione dell'algoritmo di Amanatides & Woo (1987)
// "A Fast Voxel Traversal Algorithm for Ray Tracing"
// ─────────────────────────────────────────────────────────────────────────────
inline double boundary_distance(double x,  double y,  double z,
                                 double ux, double uy, double uz,
                                 int    ix, int    iy, int    iz) {
    double s = 1.0e30;

    if (std::fabs(ux) > 1.0e-12) {
        double xb = (ux > 0) ? (ix + 1) * VOXEL_CM : ix * VOXEL_CM;
        double t  = (xb - x) / ux;
        if (t > 1.0e-10) s = std::min(s, t);
    }
    if (std::fabs(uy) > 1.0e-12) {
        double yb = (uy > 0) ? (iy + 1) * VOXEL_CM : iy * VOXEL_CM;
        double t  = (yb - y) / uy;
        if (t > 1.0e-10) s = std::min(s, t);
    }
    if (std::fabs(uz) > 1.0e-12) {
        double zb = (uz > 0) ? (iz + 1) * VOXEL_CM : iz * VOXEL_CM;
        double t  = (zb - z) / uz;
        if (t > 1.0e-10) s = std::min(s, t);
    }
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// TRASPORTO FOTONE — ciclo completo di una storia
// Gestisce fotoni primari e secondari con uno stack esplicito.
// ─────────────────────────────────────────────────────────────────────────────
void transport_photon(Particle primary,
                       const int    *phantom,
                       double       *dose,
                       Xoshiro256   &rng) {

    // Stack locale per i secondari (fotoni di annichilazione)
    // Dimensione massima: molto raramente supera 10 elementi
    Particle stack[64];
    int      stack_top = 0;
    stack[stack_top++] = primary;

    while (stack_top > 0) {

        // Estrai particella corrente dallo stack
        Particle cur = stack[--stack_top];

        // ── Loop di trasporto per questa particella ───────────────────────
        for (int step = 0; step < 100000; step++) {

            // Cutoff energetico
            if (cur.energy < ECUT) {
                // Deposita energia residua nel voxel corrente
                if (inside(cur.x, cur.y, cur.z)) {
                    int ix = vox(cur.x), iy = vox(cur.y), iz = vox(cur.z);
                    dose[phantom_idx(ix, iy, iz)] += cur.energy;
                }
                break;
            }

            // Verifica bounds
            if (!inside(cur.x, cur.y, cur.z)) break;

            // Voxel corrente
            int    ix  = vox(cur.x), iy = vox(cur.y), iz = vox(cur.z);
            int    mat = phantom[phantom_idx(ix, iy, iz)];
            double mu  = get_mu_total(cur.energy, mat);

            if (mu <= 0.0) break;   // materiale trasparente (non dovrebbe accadere)

            // ── Campiona cammino libero medio: s = -ln(ξ)/μ ──────────────
            double xi = rng();
            double s_free = -std::log(xi) / mu;

            // ── Voxel traversal: quanto possiamo andare prima del confine? ─
            double s_bnd = boundary_distance(
                cur.x, cur.y, cur.z,
                cur.ux, cur.uy, cur.uz,
                ix, iy, iz
            );

            if (s_free <= s_bnd) {
                // ── Interazione nel voxel corrente ────────────────────────

                // Sposta la particella al punto di interazione
                cur.x += cur.ux * s_free;
                cur.y += cur.uy * s_free;
                cur.z += cur.uz * s_free;

                if (!inside(cur.x, cur.y, cur.z)) break;

                // Ri-calcola voxel (potrebbe essere cambiato per step grandi)
                ix = vox(cur.x); iy = vox(cur.y); iz = vox(cur.z);
                mat = phantom[phantom_idx(ix, iy, iz)];
                int vid = phantom_idx(ix, iy, iz);

                // Seleziona tipo di interazione
                int itype = select_interaction(cur.energy, mat, rng());

                // ── FOTOELETTRICO: assorbimento totale ────────────────────
                if (itype == 0) {
                    dose[vid] += cur.energy;
                    break;
                }

                // ── COMPTON: metodo di Kahn ───────────────────────────────
                else if (itype == 1) {
                    double cos_theta, E_scatter;
                    sample_compton(cur.energy, rng, cos_theta, E_scatter);

                    // Deposita energia ceduta all'elettrone (KERMA locale)
                    double deposited = cur.energy - E_scatter;
                    if (deposited > 0.0) dose[vid] += deposited;

                    // Aggiorna energia e direzione del fotone
                    cur.energy = E_scatter;
                    double phi = 2.0 * PI * rng();
                    rotate_direction(cur.ux, cur.uy, cur.uz, cos_theta, phi);

                    if (cur.energy < ECUT) {
                        dose[vid] += cur.energy;
                        break;
                    }
                    // Continua il loop con il fotone aggiornato
                }

                // ── PRODUZIONE DI COPPIE ──────────────────────────────────
                else {
                    // Energia cinetica disponibile per elettrone e positrone
                    double E_kin = cur.energy - 2.0 * ME_C2;
                    if (E_kin > 0.0) dose[vid] += E_kin;

                    // Due fotoni di annichilazione da 0.511 MeV
                    // emessi in direzioni opposte (isotrope in prima appross.)
                    if (ME_C2 > ECUT && stack_top + 2 <= 62) {
                        double ct_a = 2.0 * rng() - 1.0;
                        double phi_a = 2.0 * PI * rng();
                        double st_a  = std::sqrt(std::max(0.0, 1.0 - ct_a * ct_a));

                        Particle ann1, ann2;
                        ann1.x = ann2.x = cur.x;
                        ann1.y = ann2.y = cur.y;
                        ann1.z = ann2.z = cur.z;
                        ann1.ux =  st_a * std::cos(phi_a);
                        ann1.uy =  st_a * std::sin(phi_a);
                        ann1.uz =  ct_a;
                        ann2.ux = -ann1.ux;
                        ann2.uy = -ann1.uy;
                        ann2.uz = -ann1.uz;
                        ann1.energy = ann2.energy = ME_C2;

                        stack[stack_top++] = ann1;
                        stack[stack_top++] = ann2;
                    }
                    break;
                }

            } else {
                // ── Voxel traversal: avanza al confine del voxel ──────────
                // Piccolo epsilon per superare il confine
                double eps = 1.0e-7;
                cur.x += cur.ux * (s_bnd + eps);
                cur.y += cur.uy * (s_bnd + eps);
                cur.z += cur.uz * (s_bnd + eps);
                // Continua il loop con il nuovo voxel
            }

        } // fine loop step
    } // fine loop stack
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {

    // ── Parametri da linea di comando ────────────────────────────────────────
    long long N     = 1000000;   // default: 1M fotoni
    int phantom_type = 0;         // 0=acqua, 1=eterogeneo
    uint64_t seed   = 42ULL;

    if (argc > 1) N            = std::atoll(argv[1]);
    if (argc > 2) phantom_type = std::atoi(argv[2]);
    if (argc > 3) seed         = (uint64_t)std::atoll(argv[3]);

    const char *phantom_label = (phantom_type == 0) ? "Acqua omogenea"
                                                     : "Acqua + Osso";

    printf("══════════════════════════════════════════════════════════════\n");
    printf("  Monte Carlo per Radioterapia — CPU Sequenziale\n");
    printf("  Angelica Porco — Matricola 264034\n");
    printf("  High Performance Computing 2025/2026\n");
    printf("  Prof. D. Talia  |  Ing. R. Cantini\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("\n  Parametri:\n");
    printf("  Phantom    : %dx%dx%d voxel  |  voxel %.0fmm  |  %.0f³ cm³\n",
           NX, NY, NZ, VOXEL_CM * 10.0, PHANTOM_CM);
    printf("  Materiale  : %s\n", phantom_label);
    printf("  N fotoni   : %lld\n", N);
    printf("  Seed       : %llu\n", (unsigned long long)seed);
    printf("  ECUT       : %.0f keV\n\n", ECUT * 1000.0);

    // ── Allocazione ──────────────────────────────────────────────────────────
    int    *phantom   = new int   [NX * NY * NZ];
    double *dose      = new double[NX * NY * NZ]();  // zero-init
    double *pdd       = new double[NZ];
    double *depths    = new double[NZ];
    double *profile   = new double[NX];
    double *positions = new double[NX];

    // ── Costruzione phantom ───────────────────────────────────────────────────
    if (phantom_type == 0)
        build_phantom_water(phantom);
    else {
        printf("  Costruzione phantom eterogeneo...\n");
        build_phantom_hetero(phantom);
    }

    // ── Oggetti simulazione ───────────────────────────────────────────────────
    Spectrum    spec;          // spettro 6MV con CDF precalcolata
    Xoshiro256  rng(seed);     // PRNG xoshiro256**

    // ── Simulazione ───────────────────────────────────────────────────────────
    printf("  Avvio simulazione...\n");
    auto t_start = std::chrono::high_resolution_clock::now();

    long long report_step = std::max(1LL, N / 20);   // report ogni 5%

    for (long long i = 0; i < N; i++) {

        // Progresso ogni 5%
        if ((i + 1) % report_step == 0) {
            auto now     = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_start).count();
            double rate  = (i + 1) / elapsed;
            double eta   = (N - i - 1) / rate;
            printf("  [%5.1f%%]  %.0f fotoni/s  ETA %.0fs\n",
                   100.0 * (i + 1) / N, rate, eta);
        }

        Particle p = sample_source(spec, rng);
        transport_photon(p, phantom, dose, rng);
    }

    auto t_end   = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // ── Statistiche ───────────────────────────────────────────────────────────
    print_dose_stats(dose, N, elapsed);

    // ── Analisi ───────────────────────────────────────────────────────────────
    compute_pdd(dose, pdd, depths);
    compute_lateral_profile(dose, profile, positions, 10.0);
    print_pdd_table(depths, pdd, phantom_label);

    // ── Output file ───────────────────────────────────────────────────────────
    printf("\n  Salvataggio output...\n");
    const char *pdd_file     = (phantom_type == 0)
                               ? "pdd_water.csv" : "pdd_hetero.csv";
    const char *prof_file    = (phantom_type == 0)
                               ? "profile_water.csv" : "profile_hetero.csv";
    const char *slice_file   = (phantom_type == 0)
                               ? "dose_slice_water.csv" : "dose_slice_hetero.csv";
    const char *bin_file     = (phantom_type == 0)
                               ? "dose_water.bin" : "dose_hetero.bin";

    save_pdd_csv(depths, pdd, pdd_file);
    save_profile_csv(positions, profile, prof_file);
    save_dose_slice_csv(dose, slice_file);
    save_dose_binary(dose, bin_file);

    // ── Pulizia ───────────────────────────────────────────────────────────────
    delete[] phantom;
    delete[] dose;
    delete[] pdd;
    delete[] depths;
    delete[] profile;
    delete[] positions;

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  Simulazione completata.\n");
    printf("  Prossimi passi:\n");
    printf("  1. Esegui tests.py per la validazione fisica\n");
    printf("  2. Aumenta N a 1e7 per accuratezza clinica\n");
    printf("  3. Confronta pdd_water.csv con DOSXYZnrc\n");
    printf("  4. Porting su CUDA → versione V1\n");
    printf("══════════════════════════════════════════════════════════════\n");

    return 0;
}
