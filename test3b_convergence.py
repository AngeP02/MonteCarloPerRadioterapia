#!/usr/bin/env python3
"""
test3b_convergence.py
─────────────────────────────────────────────────────────────────────────────
TEST 3b — Convergenza statistica: σ ∝ 1/√N

Obiettivo:
  Verificare che la deviazione standard della dose in un voxel centrale
  scala come 1/√N al variare del numero di particelle simulate.
  Se la pendenza del grafico log-log(σ vs N) è significativamente diversa
  da −0.5, il generatore casuale ha correlazioni (o c'è un bug nel trasporto).

Procedura:
  Per ogni N in {10⁴, 10⁵, 10⁶, 10⁷}:
    1. Lancia n_rep = 5 simulazioni indipendenti (seed diversi)
    2. Legge la dose nel voxel centrale (ix=50, iy=50, iz=33 → z≈10 cm)
       dal file binario dose_water.bin (formato float64, NX×NY×NZ)
    3. Calcola σ = std(dose_per_fotone) tra le repliche

  Poi esegue regressione log-log(σ vs N) e verifica:
    |pendenza − (−0.5)| < 0.10   → PASS
    |pendenza − (−0.5)| ∈ [0.10, 0.20) → WARN
    |pendenza − (−0.5)| ≥ 0.20   → FAIL

Output:
  - Tabella stdout con N, <σ>, CV% per ogni livello
  - Pendenza log-log e risultato del test
  - File PNG: test3b_sigma_vs_N.png (grafico log-log con fit)
  - File CSV: test3b_results.csv

Prerequisiti:
  - Eseguibile ./mc_rt_cpu compilato da main.cpp
  - Librerie Python: numpy, matplotlib

Uso:
  python3 test3b_convergence.py

Nota sui tempi:
  N=10⁷ richiede ~60-120 secondi per replica su CPU sequenziale.
  Con n_rep=5 il test 3b può durare 6-10 minuti totali.
  Se il tempo è un problema, ridurre n_rep a 3 (soglia statistica più larga).

Autore: Angelica Porco — Matricola 264034
Corso:  High Performance Computing 2025/2026
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import os
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────
EXECUTABLE  = "./mc_rt_cpu"      # eseguibile compilato
DOSE_FILE   = "dose_water.bin"   # file binario output della simulazione
NX = NY = NZ = 100               # dimensioni griglia (da physics.h)
VOXEL_CM    = 0.30               # lato voxel [cm]

# Voxel di misura: centrale in x,y; z≈10 cm (profondità di riferimento clinica)
IX_MEAS = NX // 2       # 50
IY_MEAS = NY // 2       # 50
IZ_MEAS = int(10.0 / VOXEL_CM)  # 33  →  z ≈ 10 cm

# N da testare (richiesti dalla traccia: 10⁴, 10⁵, 10⁶, 10⁷)
N_VALUES = [10_000, 100_000, 1_000_000, 10_000_000]
N_REP    = 5     # repliche indipendenti per ogni N

# Seed base per le repliche (replica i → seed = SEED_BASE + i)
SEED_BASE = 1000

# Soglie per il test della pendenza
SLOPE_TARGET   = -0.5
SLOPE_TOL_PASS = 0.10   # |slope − (−0.5)| < 0.10  → PASS
SLOPE_TOL_WARN = 0.20   # |slope − (−0.5)| < 0.20  → WARN, altrimenti FAIL


# ─────────────────────────────────────────────────────────────────────────────
# FUNZIONI AUSILIARIE
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(N: int, seed: int) -> float | None:
    """
    Lancia mc_rt_cpu con N fotoni e dato seed, legge dose_water.bin,
    restituisce la dose nel voxel di misura normalizzata per fotone [MeV/fotone].
    Restituisce None in caso di errore.
    """
    cmd = [EXECUTABLE, str(int(N)), "0", str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    [ERRORE] N={N:,}, seed={seed}: {result.stderr[:120]}")
        return None

    if not os.path.exists(DOSE_FILE):
        print(f"    [ERRORE] File {DOSE_FILE} non trovato dopo N={N:,}")
        return None

    # Legge griglia 3D: il file è scritto in ordine x + NX*(y + NY*z)
    # (vedi phantom_idx in physics.h: ix + NX*(iy + NY*iz))
    dose_flat = np.fromfile(DOSE_FILE, dtype=np.float64)

    if dose_flat.size != NX * NY * NZ:
        print(f"    [ERRORE] Dimensione file inattesa: {dose_flat.size}")
        return None

    dose_3d = dose_flat.reshape((NZ, NY, NX))  # attenzione all'ordine degli assi!
    # dose_flat[ix + NX*(iy + NY*iz)] → dose_3d[iz, iy, ix]

    dose_voxel = dose_3d[IZ_MEAS, IY_MEAS, IX_MEAS]

    # Normalizza per fotone per rendere confrontabili run con N diversi
    return dose_voxel / N


def print_separator():
    print("  " + "─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  TEST 3b — Convergenza statistica σ ∝ 1/√N                  ║")
    print("║  Angelica Porco — Matricola 264034                           ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Verifiche preliminari ─────────────────────────────────────────────────
    if not os.path.exists(EXECUTABLE):
        print(f"ERRORE: {EXECUTABLE} non trovato.")
        print("Compila prima con:  g++ -O2 -std=c++17 -o mc_rt_cpu main.cpp -lm")
        sys.exit(1)

    print(f"  Voxel di misura: ix={IX_MEAS}, iy={IY_MEAS}, iz={IZ_MEAS}")
    print(f"  Profondità corrispondente: z = {(IZ_MEAS + 0.5)*VOXEL_CM:.2f} cm")
    print(f"  Repliche per ogni N: {N_REP}")
    print(f"  N testati: {', '.join(f'{n:,}' for n in N_VALUES)}\n")

    sigma_list = []
    mean_list  = []
    N_ok       = []

    t_total_start = time.time()

    # ── Loop su N ─────────────────────────────────────────────────────────────
    print(f"  {'N':>12}  {'Rep OK':>6}  {'<dose/N>':>14}  {'σ':>14}  {'CV %':>8}")
    print_separator()

    for N in N_VALUES:
        doses = []
        t_n_start = time.time()

        for rep in range(N_REP):
            seed = SEED_BASE + rep
            d = run_simulation(N, seed)
            if d is not None:
                doses.append(d)

        t_n = time.time() - t_n_start

        if len(doses) < 3:
            print(f"  {N:>12,}  {'<3 OK':>6}  {'—':>14}  {'—':>14}  {'—':>8}")
            print(f"    Troppo poche repliche valide, impossibile stimare σ.")
            continue

        doses_arr = np.array(doses)
        m  = doses_arr.mean()
        s  = doses_arr.std(ddof=1)  # std campionaria (ddof=1)
        cv = (s / m * 100.0) if m > 0 else 0.0

        sigma_list.append(s)
        mean_list.append(m)
        N_ok.append(N)

        print(f"  {N:>12,}  {len(doses):>6}  {m:>14.4e}  {s:>14.4e}  {cv:>8.3f}"
              f"  [{t_n:.0f}s]")

    t_total = time.time() - t_total_start
    print_separator()
    print(f"  Tempo totale simulazioni: {t_total:.0f} s\n")

    # ── Regressione log-log ───────────────────────────────────────────────────
    if len(N_ok) < 3:
        print("ERRORE: meno di 3 livelli N disponibili, impossibile stimare pendenza.")
        sys.exit(1)

    N_arr  = np.array(N_ok, dtype=float)
    s_arr  = np.array(sigma_list)

    slope, intercept = np.polyfit(np.log10(N_arr), np.log10(s_arr), 1)

    # Linea di fit e curva teorica 1/√N
    sigma0     = s_arr[0]   # valore di riferimento al primo punto
    N_theory   = np.logspace(np.log10(N_arr.min()), np.log10(N_arr.max()), 100)
    s_theory   = sigma0 * np.sqrt(N_arr[0] / N_theory)
    s_fit      = 10**intercept * N_arr**slope

    print(f"  ── Risultati regressione log-log ──────────────────────────")
    print(f"  Pendenza stimata   : {slope:.4f}")
    print(f"  Pendenza teorica   : {SLOPE_TARGET:.4f}")
    print(f"  Scarto             : {slope - SLOPE_TARGET:+.4f}")
    print(f"  Intercetta (log10) : {intercept:.4f}  →  A = {10**intercept:.4e}")

    # Determinazione risultato
    delta = abs(slope - SLOPE_TARGET)
    if delta < SLOPE_TOL_PASS:
        result_str = "PASS ✓"
        result_note = f"(|Δ| = {delta:.3f} < {SLOPE_TOL_PASS})"
    elif delta < SLOPE_TOL_WARN:
        result_str = "WARN ⚠"
        result_note = (f"(|Δ| = {delta:.3f} ∈ [{SLOPE_TOL_PASS}, {SLOPE_TOL_WARN}))\n"
                       f"  Il generatore potrebbe avere correlazioni deboli o\n"
                       f"  le fluttuazioni con solo 5 repliche sono ancora alte.")
    else:
        result_str = "FAIL ✗"
        result_note = (f"(|Δ| = {delta:.3f} ≥ {SLOPE_TOL_WARN})\n"
                       f"  PROBLEMA: pendenza troppo diversa da -0.5.\n"
                       f"  Possibili cause: correlazioni nel PRNG, bug nel trasporto.")

    print(f"\n  ══════════════════════════════════════════════════════════")
    print(f"  Risultato TEST 3b: {result_str}  {result_note}")
    print(f"  ══════════════════════════════════════════════════════════")

    # ── Grafico log-log ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(N_arr, s_arr, 'bo', ms=9, zorder=5, label='σ misurata (5 repliche)')
    ax.loglog(N_arr, s_fit,  'b--', lw=1.5, alpha=0.7,
              label=f'Fit: σ ∝ N^{{{slope:.3f}}}')
    ax.loglog(N_theory, s_theory, 'r-', lw=2,
              label='Teorica: σ ∝ 1/√N  (pendenza = −0.500)')

    ax.set_xlabel('Numero di particelle N', fontsize=12)
    ax.set_ylabel('Deviazione standard dose [MeV/fotone]', fontsize=12)
    ax.set_title(
        f'TEST 3b — Convergenza statistica  |  pendenza = {slope:.3f}  ({result_str})',
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35, which='both')

    # Annotazioni sui punti
    for n_val, s_val in zip(N_arr, s_arr):
        ax.annotate(f'N={n_val:.0e}', (n_val, s_val),
                    textcoords='offset points', xytext=(5, 5), fontsize=8, alpha=0.7)

    plt.tight_layout()
    fig.savefig('test3b_sigma_vs_N.png', dpi=150)
    print(f"\n  Grafico salvato: test3b_sigma_vs_N.png")

    # ── Salva CSV ─────────────────────────────────────────────────────────────
    with open('test3b_results.csv', 'w') as f:
        f.write("N,mean_dose_per_photon_MeV,sigma,CV_percent\n")
        for n_val, m_val, s_val in zip(N_ok, mean_list, sigma_list):
            cv_val = s_val / m_val * 100.0 if m_val > 0 else 0.0
            f.write(f"{n_val},{m_val:.6e},{s_val:.6e},{cv_val:.4f}\n")
    print(f"  Dati salvati: test3b_results.csv")

    return 0 if delta < SLOPE_TOL_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
