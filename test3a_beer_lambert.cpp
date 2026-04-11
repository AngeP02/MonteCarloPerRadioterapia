/*
 * test3a_beer_lambert.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * TEST 3a — Verifica Beer-Lambert analitico
 *
 * Setup:
 *   - Fotoni monoenergetici 2 MeV, fascio parallelo lungo +z
 *   - Phantom acqua omogenea 30×30×30 cm³ (come main.cpp)
 *   - Solo interazione fotoelettrica: Compton e pair production disabilitati
 *     → ogni fotone viene assorbito senza scattering
 *     → la dose asse-z segue esattamente D(z) = D(0)·exp(−μ·z)
 *
 * Verifica attesa:
 *   μ_acqua @ 2 MeV = 0.04942 cm⁻¹  (NIST XCOM, Hubbell & Seltzer 1996)
 *   D(10 cm) = exp(−0.04942 × 10) × 100 = 60.8%
 *   Tolleranza: 2% assoluto su ogni punto
 *
 * Compilazione:
 *   g++ -O2 -std=c++17 -o test3a test3a_beer_lambert.cpp -lm
 * Esecuzione:
 *   ./test3a
 *
 * Output:
 *   - Tabella stdout: z | D_MC | D_analitica | diff | PASS/FAIL
 *   - File CSV: test3a_pdd_comparison.csv
 *
 * Autore: Angelica Porco — Matricola 264034
 * Corso:  High Performance Computing 2025/2026
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>

// Tutti gli header del progetto originale
#include "physics.h"
#include "random.h"
#include "phantom.h"
#include "output.h"

// ─────────────────────────────────────────────────────────────────────────────
// TRASPORTO FOTONE — SOLO FOTOELETTRICO (no Compton, no pair)
//
// Versione semplificata di transport_photon() di main.cpp:
//   - La sezione d'urto usata è mu_total (= l'attenuazione totale del mezzo)
//   - A ogni interazione il fotone viene assorbito completamente
//   - Nessun fotone secondario generato
//
// Perché funziona per Beer-Lambert:
//   La fluenza del fascio decade come Φ(z) = Φ₀·exp(−μz).
//   Con deposito locale (no scatter), la dose riproduce esattamente la fluenza.
//
// Parametri:
//   px, py    : posizione x,y di ingresso nel phantom [cm]
//   energy    : energia fotone [MeV]
//   dose      : griglia dose NX×NY×NZ [MeV per voxel]
//   rng       : generatore xoshiro256**
// ─────────────────────────────────────────────────────────────────────────────
static void transport_pe_only(double px, double py,
                               double energy,
                               double *dose,
                               Xoshiro256 &rng)
{
    // Fascio parallelo a +z, parte appena dentro la superficie superiore
    double x = px, y = py, z = 1.0e-7;
    const double ux = 0.0, uy = 0.0, uz = 1.0;

    // Coefficiente di attenuazione lineare [cm⁻¹] = (μ/ρ) × ρ
    const double mu = get_mu_total(energy, MAT_WATER);

    for (int step = 0; step < 100000; step++) {

        if (!inside(x, y, z)) break;
        if (energy < ECUT)    break;

        int ix = vox(x), iy = vox(y), iz = vox(z);

        // Distanza al confine del voxel corrente lungo z (uz = 1)
        double z_boundary = (iz + 1) * VOXEL_CM;
        double s_boundary = z_boundary - z;   // uz = 1, quindi s = Δz
        if (s_boundary < 1.0e-10) s_boundary = 1.0e-10;

        // Campiona il cammino libero medio: s = −ln(ξ) / μ
        double xi     = rng();
        double s_free = -std::log(xi) / mu;

        if (s_free <= s_boundary) {
            // ── Assorbimento fotoelettrico nel voxel corrente ─────────────
            x += ux * s_free;
            y += uy * s_free;
            z += uz * s_free;

            if (!inside(x, y, z)) break;

            // Deposita tutta l'energia nel voxel di assorbimento
            int iix = vox(x), iiy = vox(y), iiz = vox(z);
            dose[phantom_idx(iix, iiy, iiz)] += energy;
            break;  // fotone assorbito, storia terminata

        } else {
            // ── Voxel traversal: avanza al confine ───────────────────────
            x += ux * (s_boundary + 1.0e-7);
            y += uy * (s_boundary + 1.0e-7);
            z += uz * (s_boundary + 1.0e-7);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  TEST 3a — Beer-Lambert analitico\n");
    printf("  Angelica Porco — Matricola 264034\n");
    printf("══════════════════════════════════════════════════════════════\n\n");

    // ── Parametri fisici ─────────────────────────────────────────────────────
    const double E_MEV      = 2.0;     // MeV — fotoni monoenergetici
    const double MU_NIST    = 0.04942; // cm⁻¹ — NIST XCOM acqua @ 2 MeV
    const double TOLERANCE  = 2.0;     // % assoluto — tolleranza richiesta
    const long long N       = 5000000; // fotoni simulati (5M per rumore < 0.3%)
    const double FIELD_HALF = 5.0;     // ±5 cm → campo 10×10 cm²
    const double cx         = PHANTOM_CM / 2.0;
    const double cy         = PHANTOM_CM / 2.0;

    // ── 1. Verifica coerenza tabella interna con NIST ─────────────────────────
    double mu_interp = get_mu_total(E_MEV, MAT_WATER);
    printf("  [1] Coefficiente di attenuazione μ a 2 MeV in acqua:\n");
    printf("      Tabella interna (interpolata) : %.5f cm⁻¹\n", mu_interp);
    printf("      NIST XCOM (riferimento)        : %.5f cm⁻¹\n", MU_NIST);
    printf("      Discrepanza                    : %.4f%%\n\n",
           std::fabs(mu_interp - MU_NIST) / MU_NIST * 100.0);

    // ── 2. Calcolo a mano Beer-Lambert a 10 cm ───────────────────────────────
    double D10_anal = std::exp(-MU_NIST * 10.0) * 100.0;
    printf("  [2] Calcolo a mano — D(z) = 100 × exp(−μz):\n");
    printf("      D(10 cm) = 100 × exp(−%.5f × 10)\n", MU_NIST);
    printf("               = 100 × exp(−%.5f)\n", MU_NIST * 10.0);
    printf("               = 100 × %.6f\n", std::exp(-MU_NIST * 10.0));
    printf("               = %.3f%%   (atteso: ~60.8%%)\n\n", D10_anal);

    // ── 3. Allocazione e costruzione phantom ─────────────────────────────────
    int    *phantom = new int   [NX * NY * NZ];
    double *dose    = new double[NX * NY * NZ]();
    build_phantom_water(phantom);

    Xoshiro256 rng(42ULL);

    // ── 4. Simulazione ────────────────────────────────────────────────────────
    printf("  [3] Simulazione: N = %lld fotoni monoenergetici %.1f MeV...\n",
           N, E_MEV);
    for (long long i = 0; i < N; i++) {
        double px = cx + (rng() * 2.0 - 1.0) * FIELD_HALF;
        double py = cy + (rng() * 2.0 - 1.0) * FIELD_HALF;
        transport_pe_only(px, py, E_MEV, dose, rng);
    }
    printf("      Completato.\n\n");

    // ── 5. Calcolo PDD ────────────────────────────────────────────────────────
    double pdd[NZ], depths[NZ];
    compute_pdd(dose, pdd, depths);  // normalizza al massimo → [0, 100%]

    // ── 6. Tabella comparativa MC vs analitico ────────────────────────────────
    printf("  [4] Confronto MC vs Beer-Lambert — D(z) = 100·exp(−%.5f·z):\n\n",
           MU_NIST);
    printf("  %-8s  %-10s  %-14s  %-12s  %-8s\n",
           "z [cm]", "D_MC [%]", "D_analitic [%]", "Diff abs [%]", "Stato");
    printf("  %s\n",
           "──────────────────────────────────────────────────────────────");

    double z_check[] = { 1.5, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0 };
    bool all_pass = true;

    for (int r = 0; r < 7; r++) {
        double z    = z_check[r];
        int    k    = (int)(z / VOXEL_CM);
        if (k >= NZ) k = NZ - 1;

        double D_mc   = pdd[k];
        double D_anal = 100.0 * std::exp(-MU_NIST * z);
        double diff   = std::fabs(D_mc - D_anal);
        bool   ok     = (diff < TOLERANCE);
        if (!ok) all_pass = false;

        printf("  %-8.1f  %-10.2f  %-14.2f  %-12.2f  %s%s\n",
               z, D_mc, D_anal, diff,
               ok ? "PASS ✓" : "FAIL ✗",
               (z == 10.0) ? "  ← punto chiave" : "");
    }

    // ── 7. Riepilogo punto chiave a 10 cm ─────────────────────────────────────
    int    k10   = (int)(10.0 / VOXEL_CM);
    double D10mc = pdd[k10];
    printf("\n  ── Verifica chiave a 10 cm ───────────────────────────────\n");
    printf("  D(10) analitico : %.3f%%  (= exp(−0.04942×10)×100)\n", D10_anal);
    printf("  D(10) MC        : %.3f%%\n", D10mc);
    printf("  Differenza      : %.3f%%  (tolleranza: %.1f%%)\n",
           std::fabs(D10mc - D10_anal), TOLERANCE);

    // ── 8. Salva CSV per il plot Python ──────────────────────────────────────
    {
        std::ofstream f("test3a_pdd_comparison.csv");
        f << "depth_cm,D_MC_percent,D_analytical_percent,diff_abs_percent\n";
        for (int iz = 0; iz < NZ; iz++) {
            double D_a = 100.0 * std::exp(-MU_NIST * depths[iz]);
            f << depths[iz] << ","
              << pdd[iz]    << ","
              << D_a        << ","
              << std::fabs(pdd[iz] - D_a) << "\n";
        }
        printf("\n  CSV salvato: test3a_pdd_comparison.csv\n");
    }

    // ── 9. Risultato finale ───────────────────────────────────────────────────
    printf("\n  ══════════════════════════════════════════════════════════\n");
    printf("  Risultato TEST 3a: %s\n", all_pass ? "PASS ✓" : "FAIL ✗");
    if (!all_pass) {
        printf("  Nota: se un singolo punto fallisce di pochissimo (<0.1%%\n");
        printf("        oltre soglia), aumentare N a 10M o cambiare seed.\n");
        printf("        La fluttuazione statistica a 25 cm con 5M fotoni\n");
        printf("        è circa 0.4%%, quindi può capitare per seed sfortuna.\n");
    }
    printf("  ══════════════════════════════════════════════════════════\n");

    delete[] phantom;
    delete[] dose;
    return all_pass ? 0 : 1;
}
