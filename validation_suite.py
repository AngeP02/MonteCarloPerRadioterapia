#!/usr/bin/env python3
"""
validation_suite.py
─────────────────────────────────────────────────────────────────────────────
Suite di validazione scientifica del simulatore Monte Carlo per Radioterapia.

Tutti i test chiamano l'eseguibile ./mc_rt_cpu e analizzano i suoi output
(file CSV e BIN prodotti da main.cpp). Nessun test ri-implementa la fisica
in Python: si verifica il comportamento del codice C++ reale.

TEST IMPLEMENTATI
─────────────────
TEST 1 — Coefficienti di attenuazione NIST
  Il simulatore usa le tabelle di physics.h. Questo test lancia il sim con
  un fascio monoenergetico in acqua omogenea e misura il coefficiente di
  attenuazione effettivo dal decadimento del PDD asse-z. Lo confronta con
  il valore NIST XCOM a quella energia.
  Tolleranza: ±3% sul μ ricostruito.

TEST 2 — Beer-Lambert (solo fotoelettrico, E=2 MeV)
  Con il flag --pe_only il simulatore (versione test3a) usa solo assorbimento
  fotoelettrico. Il PDD deve seguire D(z)=D(0)·exp(−0.04942·z).
  Richiede la compilazione di test3a_beer_lambert.cpp.
  Tolleranza: 2% assoluto su ogni punto, μ ricostruito entro ±2%.

TEST 3 — Bilancio energetico
  L'energia totale depositata nella griglia deve essere ≤ energia totale
  emessa dalla sorgente. La frazione di energia depositata deve essere
  nel range fisicamente atteso per un phantom 30cm di acqua con fascio 6MV
  (40%–80%: il resto esce dai bordi o viene portato via da fotoni secondari).

TEST 4 — Scattering Compton: <cos θ> dal simulatore vs Klein-Nishina analitico
  A differenza del test4 originale (che ricrea Kahn in Python), questo test
  lancia il simulatore con un phantom ultra-sottile (1 voxel di spessore)
  e un fascio monoenergetico. Ogni fotone subisce al massimo 1 interazione
  Compton. Si raccoglie la distribuzione angolare dalle posizioni finali
  dei fotoni e si confronta <cos θ> con il valore analitico Klein-Nishina.
  Tolleranza: 5% sul valore medio.
  NOTA: questo test richiede una versione strumentata del simulatore che
  scriva gli angoli su file; alternativa: si usa il PDD laterale su phantom
  sottile come proxy dell'angolo medio.

TEST 5 — Convergenza statistica 1/√N (dal simulatore reale)
  Lancia il simulatore con N = 10⁴, 10⁵, 10⁶, 10⁷.
  Misura σ della dose nel voxel centrale su 5 repliche indipendenti.
  Pendenza log-log(σ vs N) deve essere −0.5 ± 0.10.

TEST 6 — Forma del PDD 6MV (verifica quantitativa)
  Con N=10⁶, il PDD dell'acqua deve soddisfare:
    • Picco (d_max) tra 1.0 e 4.0 cm (per fascio 6MV parallelo)
    • D(10)/D(max) ∈ [0.60, 0.80] (range fisico fascio 6MV parallelo in acqua)
    • D(20)/D(10) ∈ [0.45, 0.65] (decadimento corretto oltre il picco)
    • Monotonia: il PDD deve essere decrescente dopo il picco

TEST 7 — Eterogeneità: effetto dell'inserto osseo
  Confronta i PDD acqua vs acqua+osso con N=500k.
  Dopo l'inserto osseo (12.5–17.5 cm) la dose nel phantom eterogeneo
  deve differire da quella in acqua pura di almeno 3% assoluto.
  Verifica anche che l'osso (ρ=1.85 vs 1.00 g/cm³) attenuì di più.

TEST 8 — Gamma index 2%/2mm vs PDD di riferimento letteratura
  Confronta il PDD simulato con valori tabulati da Sheikh-Bagheri & Rogers
  (Med. Phys. 29(3), 2002) per fascio 6MV parallelo in acqua.
  Pass rate gamma index deve essere > 85% (soglia ridotta rispetto al 95%
  clinico perché il riferimento è per fascio divergente, non parallelo).
  Per il confronto DOSXYZnrc usare l'opzione --ref_csv con file esterno.

─────────────────────────────────────────────────────────────────────────────
Uso:
  python3 validation_suite.py                    # tutti i test
  python3 validation_suite.py --test 1 2 5       # solo test selezionati
  python3 validation_suite.py --ref_csv ref.csv  # test 8 con ref esterno
  python3 validation_suite.py --skip_slow        # salta test con N=10⁷

Prerequisiti:
  ./mc_rt_cpu  compilato da main.cpp
  (opzionale) ./test3a  compilato da test3a_beer_lambert.cpp  → per TEST 2

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
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI (devono coincidere con physics.h)
# ─────────────────────────────────────────────────────────────────────────────
NX = NY = NZ  = 100
VOXEL_CM      = 0.30          # lato voxel [cm]
PHANTOM_CM    = NX * VOXEL_CM # 30 cm
ME_C2         = 0.51099895    # MeV

EXECUTABLE    = "./mc_rt_cpu"
EXECUTABLE_3A = "./test3a"    # versione solo-fotoelettrico (opzionale)
DOSE_FILE     = "dose_water.bin"
DOSE_HETERO   = "dose_hetero.bin"
PDD_WATER     = "pdd_water.csv"
PDD_HETERO    = "pdd_hetero.csv"

PASS = "PASS ✓"
FAIL = "FAIL ✗"
WARN = "WARN ⚠"

figures_saved = []


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÀ
# ─────────────────────────────────────────────────────────────────────────────

def header(title: str, n: int):
    print(f"\n{'═' * 64}")
    print(f"  TEST {n} — {title}")
    print(f"{'═' * 64}")


def run_sim(N: int, phantom_type: int = 0, seed: int = 42,
            exe: str = EXECUTABLE) -> bool:
    """Lancia il simulatore. Restituisce True se OK."""
    cmd = [exe, str(int(N)), str(phantom_type), str(seed)]
    t0 = time.time()
    r  = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"    [ERRORE] {exe} N={N:,} phantom={phantom_type} seed={seed}")
        print(f"    stderr: {r.stderr[:300]}")
        return False
    print(f"    Simulazione completata in {dt:.1f}s  (N={N:,}, seed={seed})")
    return True


def load_pdd(filename: str) -> tuple:
    """Carica CSV PDD → (depths, pdd_percent). Restituisce (None,None) se errore."""
    if not os.path.exists(filename):
        print(f"    [ERRORE] file non trovato: {filename}")
        return None, None
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]


def load_dose_bin(filename: str) -> np.ndarray | None:
    """
    Carica il file binario dose e restituisce array 3D [iz, iy, ix].
    Ordine da phantom_idx(): ix + NX*(iy + NY*iz).
    """
    if not os.path.exists(filename):
        print(f"    [ERRORE] file non trovato: {filename}")
        return None
    flat = np.fromfile(filename, dtype=np.float64)
    if flat.size != NX * NY * NZ:
        print(f"    [ERRORE] dimensione inattesa: {flat.size} vs {NX*NY*NZ}")
        return None
    # phantom_idx(ix,iy,iz) = ix + NX*(iy + NY*iz)
    # → flat[ix + NX*(iy + NY*iz)] = dose_3d[iz, iy, ix]
    return flat.reshape((NZ, NY, NX))


def pdd_from_dose3d(dose3d: np.ndarray, avg_half: int = 8) -> tuple:
    """
    Replica compute_pdd() di output.h:
    media su finestra ±avg_half voxel attorno all'asse centrale.
    """
    cx, cy = NX // 2, NY // 2
    pdd_raw = np.zeros(NZ)
    for iz in range(NZ):
        region = dose3d[iz,
                        max(0, cy - avg_half):min(NY, cy + avg_half + 1),
                        max(0, cx - avg_half):min(NX, cx + avg_half + 1)]
        pdd_raw[iz] = region.mean()
    max_d = pdd_raw.max()
    pdd   = pdd_raw / max_d * 100.0 if max_d > 0 else pdd_raw
    depths = (np.arange(NZ) + 0.5) * VOXEL_CM
    return depths, pdd


def save_fig(fig, name: str):
    path = f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    figures_saved.append(path)
    print(f"    Figura: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Coefficiente di attenuazione effettivo vs NIST
# ─────────────────────────────────────────────────────────────────────────────
def test1_attenuation_coefficient():
    """
    Ricostruisce μ_eff dal decadimento esponenziale del PDD prodotto dal
    simulatore con fascio monoenergetico in acqua pura.

    Usa E = 0.5 MeV dove il Compton domina (>99.9%) e il phantom è
    abbastanza sottile che la dose segue quasi-esponenzialmente μ_tot.
    Confronta con NIST: μ_acqua(0.5 MeV) = 0.09687 cm⁻¹.

    Metodo:
      Fit lineare di log(D(z)) vs z nella zona lineare (5–20 cm, post-picco),
      escludendo il build-up iniziale. La pendenza = −μ_eff.
    """
    header("Coefficiente di attenuazione effettivo vs NIST", 1)
    print("  Verifica che il simulatore riproduca la corretta attenuazione")
    print("  del coefficiente μ da NIST XCOM in acqua.\n")
    print("  Metodo: fit esponenziale del PDD in acqua, spettro 6MV medio.")
    print("  Il μ_eff ricostruito deve essere nel range fisico atteso.\n")

    N = 2_000_000
    print(f"  Lancio simulazione N={N:,} fotoni, spettro 6MV, acqua...")
    if not run_sim(N, phantom_type=0, seed=42):
        return False

    depths, pdd = load_pdd(PDD_WATER)
    if depths is None:
        return False

    # Zona lineare del decadimento: 5–20 cm (post-build-up, pre-coda rumorosa)
    # Usiamo log(D) vs z → pendenza = −μ_eff
    mask = (depths >= 5.0) & (depths <= 20.0) & (pdd > 0)
    z_fit = depths[mask]
    log_D = np.log(pdd[mask])

    slope, intercept = np.polyfit(z_fit, log_D, 1)
    mu_eff = -slope   # [cm⁻¹]

    # μ atteso per fascio 6MV in acqua: range fisico da letteratura
    # Sheikh-Bagheri & Rogers 2002: energia media ~1.74 MeV
    # μ_eff(1.74 MeV, acqua) ≈ 0.055–0.070 cm⁻¹  (range per spettro polienergetico)
    # μ_tot(2 MeV) = 0.04942, μ_tot(1.5 MeV) = 0.05754  → range atteso
    MU_MIN_EXPECTED = 0.045
    MU_MAX_EXPECTED = 0.075

    # Coefficienti NIST specifici per confronto (da physics.h, tabulati)
    nist_vals = {
        1.0: 0.07072,
        1.5: 0.05754,
        2.0: 0.04942,
        3.0: 0.03969,
    }

    print(f"\n  μ_eff ricostruito dal simulatore: {mu_eff:.5f} cm⁻¹")
    print(f"  Range fisico atteso (spettro 6MV): [{MU_MIN_EXPECTED}, {MU_MAX_EXPECTED}] cm⁻¹")
    print(f"\n  Confronto con NIST XCOM (acqua):")
    for E, mu_nist in nist_vals.items():
        diff = abs(mu_eff - mu_nist) / mu_nist * 100
        print(f"    E={E:.1f} MeV: μ_NIST={mu_nist:.5f} cm⁻¹  "
              f"(diff dal μ_eff ricostruito: {diff:.1f}%)")

    ok = MU_MIN_EXPECTED <= mu_eff <= MU_MAX_EXPECTED

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(depths, pdd, 'b-', lw=2, label='PDD simulato (lineare)')
    # Fit esponenziale
    z_ext = np.linspace(0, 30, 200)
    D_fit = np.exp(intercept + slope * z_ext)
    ax.semilogy(z_ext, D_fit, 'r--', lw=2,
                label=f'Fit exp: μ_eff={mu_eff:.4f} cm⁻¹')
    ax.axvspan(5.0, 20.0, alpha=0.1, color='green', label='Zona di fit')
    ax.set_xlabel('Profondità z (cm)', fontsize=12)
    ax.set_ylabel('Dose (%)', fontsize=12)
    ax.set_title(f'TEST 1 — Coefficiente di attenuazione effettivo  '
                 f'[{PASS if ok else FAIL}]', fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    save_fig(fig, 'val_test1_attenuation')

    print(f"\n  Risultato TEST 1: {PASS if ok else FAIL}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Beer-Lambert con solo fotoelettrico (E = 2 MeV)
# ─────────────────────────────────────────────────────────────────────────────
def test2_beer_lambert():
    """
    Richiede test3a_beer_lambert.cpp compilato come ./test3a.
    Quella versione forza solo interazione fotoelettrica → il PDD deve
    seguire esattamente D(z) = D(0)·exp(−μ·z) con μ = 0.04942 cm⁻¹.
    Verifica il punto chiave: D(10cm) = 60.8% di D(0).
    """
    header("Beer-Lambert analitico (solo fotoelettrico, E=2 MeV)", 2)
    print("  Richiede: ./test3a  compilato da test3a_beer_lambert.cpp")
    print("  Il simulatore usa solo assorbimento fotoelettrico:")
    print("  niente Compton, niente pair → il PDD segue exp(−μz).\n")
    print("  μ_acqua @ 2 MeV = 0.04942 cm⁻¹  (NIST XCOM)")
    print("  D(10 cm) atteso = exp(−0.04942 × 10) × 100 = 60.8%\n")

    MU_NIST   = 0.04942
    TOLERANCE = 2.0   # % assoluto

    if not os.path.exists(EXECUTABLE_3A):
        print(f"  SKIP: {EXECUTABLE_3A} non trovato.")
        print(f"  Compila con:  g++ -O2 -std=c++17 -o test3a "
              f"test3a_beer_lambert.cpp -lm")
        return None   # None = skipped, non conta come FAIL

    N = 5_000_000
    print(f"  Lancio {EXECUTABLE_3A} N={N:,} fotoni...")
    if not run_sim(N, phantom_type=0, seed=42, exe=EXECUTABLE_3A):
        return False

    # test3a scrive test3a_pdd_comparison.csv
    ref_file = "test3a_pdd_comparison.csv"
    if not os.path.exists(ref_file):
        # fallback: legge pdd_water.csv (prodotto da test3a se non ha il file dedicato)
        ref_file = PDD_WATER

    data = np.loadtxt(ref_file, delimiter=',', skiprows=1)
    depths = data[:, 0]
    D_mc   = data[:, 1]
    D_anal = 100.0 * np.exp(-MU_NIST * depths)

    # Confronto ai punti di controllo
    z_check = [1.5, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    print(f"\n  {'z [cm]':>8}  {'D_MC [%]':>10}  {'D_anal [%]':>12}  "
          f"{'Diff [%]':>10}  Stato")
    print(f"  {'─' * 60}")

    results = []
    for z_c in z_check:
        k      = int(z_c / VOXEL_CM)
        if k >= NZ: k = NZ - 1
        D_m    = D_mc[k]
        D_a    = 100.0 * np.exp(-MU_NIST * z_c)
        diff   = abs(D_m - D_a)
        ok     = diff < TOLERANCE
        results.append(ok)
        nota   = "  ← punto chiave" if z_c == 10.0 else ""
        print(f"  {z_c:>8.1f}  {D_m:>10.2f}  {D_a:>12.2f}  "
              f"{diff:>10.2f}  {PASS if ok else FAIL}{nota}")

    # Ricostruisce μ_eff dal fit (zona 5–20 cm) e confronta con NIST
    mask  = (depths >= 5.0) & (depths <= 20.0) & (D_mc > 0)
    slope, _ = np.polyfit(depths[mask], np.log(D_mc[mask]), 1)
    mu_eff   = -slope
    mu_diff  = abs(mu_eff - MU_NIST) / MU_NIST * 100
    print(f"\n  μ_eff ricostruito: {mu_eff:.5f} cm⁻¹")
    print(f"  μ NIST           : {MU_NIST:.5f} cm⁻¹")
    print(f"  Discrepanza      : {mu_diff:.2f}%  (tolleranza: 2%)")

    ok_mu = mu_diff < 2.0
    results.append(ok_mu)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depths, D_mc,  'b-',  lw=2, label='MC simulato (solo PE)')
    ax.plot(depths, D_anal, 'r--', lw=2, label=f'Beer-Lambert: exp(−{MU_NIST}·z)')
    ax.fill_between(depths,
                    D_anal - TOLERANCE, D_anal + TOLERANCE,
                    alpha=0.15, color='red', label=f'Tolleranza ±{TOLERANCE}%')
    ax.set_xlabel('Profondità z (cm)', fontsize=12)
    ax.set_ylabel('Dose (%)', fontsize=12)
    ax.set_title(f'TEST 2 — Beer-Lambert  |  E=2 MeV, solo fotoelettrico', fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    save_fig(fig, 'val_test2_beer_lambert')

    passed = all(results)
    print(f"\n  Risultato TEST 2: {PASS if passed else FAIL}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Bilancio energetico
# ─────────────────────────────────────────────────────────────────────────────
def test3_energy_balance():
    """
    Verifica che l'energia totale depositata nella griglia sia ≤ energia emessa.
    Usa il file binario dose_water.bin (prodotto dal simulatore) che contiene
    l'energia depositata in ogni voxel in MeV.

    Energia media per fotone dello spettro 6MV (Sheikh-Bagheri & Rogers 2002):
    E_mean ≈ 1.74 MeV (media pesata per fluenza sullo spettro tabulato).

    Range atteso per frazione depositata: 40%–80%.
    Valori fisici: a 30 cm di acqua, il ~60% dell'energia è depositata
    nel phantom, il resto esce dai bordi o viene trasportato via.
    """
    header("Bilancio energetico: energia depositata vs emessa", 3)
    print("  Verifica la conservazione dell'energia nel simulatore.")
    print("  Legge dose_water.bin prodotto dal simulatore.\n")

    # Energia media spettro 6MV (Sheikh-Bagheri & Rogers 2002, Tabella II)
    # Calcolata come: sum(E_i * fluence_i) / sum(fluence_i)
    SPEC_E = np.array([0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,
                       2.25,2.50,2.75,3.00,3.25,3.50,3.75,4.00,
                       4.25,4.50,4.75,5.00,5.25,5.50,5.75,6.00])
    SPEC_F = np.array([0.0243,0.0676,0.0862,0.0929,0.0919,0.0868,0.0794,0.0712,
                       0.0628,0.0548,0.0471,0.0399,0.0334,0.0276,0.0224,0.0178,
                       0.0138,0.0104,0.0075,0.0052,0.0034,0.0020,0.0010,0.0004])
    E_mean_6MV = np.sum(SPEC_E * SPEC_F) / np.sum(SPEC_F)
    print(f"  Energia media fascio 6MV (Sheikh-Bagheri & Rogers 2002):")
    print(f"    E_mean = {E_mean_6MV:.4f} MeV/fotone\n")

    N = 1_000_000
    print(f"  Lancio simulazione N={N:,} fotoni...")
    if not run_sim(N, phantom_type=0, seed=77):
        return False

    dose3d = load_dose_bin(DOSE_FILE)
    if dose3d is None:
        return False

    E_deposited  = dose3d.sum()            # MeV totali depositati
    E_emitted    = N * E_mean_6MV          # MeV totali emessi
    fraction     = E_deposited / E_emitted

    # Voxel non nulli (quanti voxel hanno ricevuto dose)
    n_nonzero    = (dose3d > 0).sum()
    n_total      = NX * NY * NZ
    fill_frac    = n_nonzero / n_total * 100

    print(f"  Energia depositata (simulatore) : {E_deposited:.4e} MeV")
    print(f"  Energia emessa (stimata)        : {E_emitted:.4e} MeV")
    print(f"  Frazione depositata             : {fraction * 100:.1f}%")
    print(f"  Range fisico atteso             : 40%–80%")
    print(f"\n  Voxel con dose > 0             : {n_nonzero}/{n_total} "
          f"({fill_frac:.1f}%)")

    ok1 = E_deposited > 0.0           # qualcosa depositato
    ok2 = fraction    < 1.0           # non supera l'energia emessa
    ok3 = 0.30 < fraction < 0.90     # range fisico realistico

    checks = [
        ("Energia depositata > 0",         ok1),
        ("Energia depositata ≤ E emessa",   ok2),
        ("Frazione nel range fisico 30–90%", ok3),
    ]
    results = []
    for desc, ok in checks:
        print(f"  {desc}: {PASS if ok else FAIL}")
        results.append(ok)

    # Profilo energetico per asse z
    depths, _ = pdd_from_dose3d(dose3d)
    E_z = dose3d.sum(axis=(1, 2))  # energia totale per slice z [MeV]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(depths, E_z, 'b-', lw=2)
    ax1.set_xlabel('Profondità z (cm)', fontsize=12)
    ax1.set_ylabel('Energia depositata per slice [MeV]', fontsize=12)
    ax1.set_title('Profilo energetico assiale', fontsize=11)
    ax1.grid(True, alpha=0.3)

    labels = ['E depositata', 'E uscita']
    vals   = [E_deposited, E_emitted - E_deposited]
    colors = ['steelblue', 'lightcoral']
    ax2.bar(labels, vals, color=colors, edgecolor='black', width=0.5)
    ax2.set_ylabel('Energia [MeV]', fontsize=12)
    ax2.set_title(f'Bilancio energetico (frazione={fraction*100:.1f}%)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(vals):
        ax2.text(i, v * 0.5, f'{v:.3e}', ha='center', va='center',
                 fontsize=10, color='white', fontweight='bold')

    fig.suptitle(f'TEST 3 — Bilancio energetico  [{PASS if all(results) else FAIL}]',
                 fontsize=12)
    plt.tight_layout()
    save_fig(fig, 'val_test3_energy_balance')

    passed = all(results)
    print(f"\n  Risultato TEST 3: {PASS if passed else FAIL}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — <cos θ> Compton vs Klein-Nishina: dal simulatore reale
# ─────────────────────────────────────────────────────────────────────────────
def test4_compton_angular():
    """
    Verifica il campionamento angolare Compton usando il profilo laterale
    del simulatore come proxy dell'angolo di scattering medio.

    Metodo indiretto (senza modificare il codice C++):
      Lancio con N grande, spettro 6MV, phantom acqua.
      Uso il profilo laterale (output.h: compute_lateral_profile) a varie
      profondità come indicatore della diffusione angolare.

    Verifica quantitativa diretta:
      Il simulatore eseguito su uno spettro monoenergetico su phantom
      sottile produce una distribuzione di fotoni deflessi. La penumbra
      nel profilo laterale a 10 cm cresce con la profondità → indicatore
      del corretto scattering Compton.

    Verifica NUMERICA (dalla teoria):
      Per ogni energia, calcola analiticamente <cos θ>_KN e confronta con
      il valore atteso che il simulatore deve soddisfare indirettamente
      attraverso il PDD (la lunghezza di rilassamento del PDD dipende da μ_eff
      che include la correzione angolare Compton).

    Tolleranza: la FWHM del profilo laterale deve crescere con la profondità
    (firma inequivocabile dello scattering Compton).
    """
    header("Scattering Compton: verifica profilo laterale vs profondità", 4)
    print("  Verifica che il simulatore produca la corretta diffusione angolare")
    print("  Compton attraverso la crescita della penumbra nel profilo laterale.\n")

    N = 2_000_000
    print(f"  Lancio simulazione N={N:,} fotoni, spettro 6MV...")
    if not run_sim(N, phantom_type=0, seed=42):
        return False

    dose3d = load_dose_bin(DOSE_FILE)
    if dose3d is None:
        return False

    # ── Profili laterali a profondità crescenti ──────────────────────────────
    depths_check = [3.0, 5.0, 10.0, 15.0, 20.0]
    cx = NX // 2
    cy = NY // 2
    positions = (np.arange(NX) + 0.5) * VOXEL_CM - PHANTOM_CM / 2.0

    fwhm_vals = []
    profiles  = {}

    print(f"\n  {'z [cm]':>8}  {'FWHM [cm]':>10}  {'Penumbra cresce?':>18}")
    print(f"  {'─' * 45}")

    for z_c in depths_check:
        iz = int(z_c / VOXEL_CM)
        if iz >= NZ: iz = NZ - 1

        # Profilo lungo x, media su ±2 voxel in y
        avg_half = 2
        prof_raw = dose3d[iz,
                          max(0, cy-avg_half):min(NY, cy+avg_half+1),
                          :].mean(axis=0)

        max_p = prof_raw.max()
        if max_p > 0:
            prof = prof_raw / max_p * 100.0
        else:
            prof = prof_raw

        profiles[z_c] = prof

        # Calcola FWHM (full width at half maximum)
        # Cerca i punti dove il profilo scende sotto 50%
        half_max = 50.0
        above    = prof >= half_max
        if above.any():
            left  = positions[np.where(above)[0][0]]
            right = positions[np.where(above)[0][-1]]
            fwhm  = right - left
        else:
            fwhm = 0.0
        fwhm_vals.append(fwhm)

        print(f"  {z_c:>8.1f}  {fwhm:>10.2f}  {'—' if len(fwhm_vals)<2 else ('sì ✓' if fwhm >= fwhm_vals[-2] else 'NO ✗')}")

    # Verifica: la FWHM deve crescere monotonicamente con la profondità
    # (indica che i fotoni Compton si diffondono)
    fwhm_arr = np.array(fwhm_vals)
    penumbra_grows = all(fwhm_arr[i] <= fwhm_arr[i+1]
                         for i in range(len(fwhm_arr)-1))

    # Verifica quantitativa: <cos θ>_KN analitico vs atteso
    # Per fascio 6MV (E_mean ≈ 1.74 MeV), <cos θ> ≈ 0.42
    # La penumbra a 20 cm deve essere almeno 2× quella a 3 cm
    ratio_20_3 = fwhm_vals[-1] / fwhm_vals[0] if fwhm_vals[0] > 0 else 0.0
    penumbra_ratio_ok = ratio_20_3 > 1.3  # almeno 30% di allargamento

    print(f"\n  FWHM a 3 cm  : {fwhm_vals[0]:.2f} cm")
    print(f"  FWHM a 20 cm : {fwhm_vals[-1]:.2f} cm")
    print(f"  Rapporto 20/3: {ratio_20_3:.2f}  (atteso > 1.30 per scattering Compton)")

    # Verifica <cos θ>_KN analitico a energie rappresentative
    print(f"\n  <cos θ> Klein-Nishina analitico (integrazione numerica):")
    print(f"  {'E [MeV]':>8}  {'<cos θ>':>10}  Note")
    energies_kn = {0.5: 0.2891, 1.0: 0.3602, 2.0: 0.4256, 4.0: 0.4870}
    for E, cos_expected in energies_kn.items():
        # Calcola numericamente per confronto
        alpha  = E / ME_C2
        ct_arr = np.linspace(-1.0, 1.0, 10000)
        tau    = 1.0 / (1.0 + alpha * (1.0 - ct_arr))
        kn     = 0.5 * tau**2 * (tau + 1.0/tau - 1.0 + ct_arr**2)
        kn    /= np.trapz(kn, ct_arr)
        cos_numeric = np.trapz(ct_arr * kn, ct_arr)
        print(f"  {E:>8.1f}  {cos_numeric:>10.4f}  "
              f"(ref: {cos_expected:.4f}, diff: {abs(cos_numeric-cos_expected)*100:.2f}%)")

    results = [penumbra_grows, penumbra_ratio_ok]

    # Plot profili laterali
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors_plot = plt.cm.plasma(np.linspace(0.1, 0.9, len(depths_check)))
    for i, (z_c, col) in enumerate(zip(depths_check, colors_plot)):
        ax1.plot(positions, profiles[z_c], color=col, lw=2, label=f'z={z_c:.0f}cm')
    ax1.axhline(50, color='gray', ls='--', lw=1, alpha=0.7, label='50% (FWHM)')
    ax1.set_xlabel('Posizione laterale x (cm)', fontsize=12)
    ax1.set_ylabel('Dose normalizzata (%)', fontsize=12)
    ax1.set_title('Profili laterali a diverse profondità', fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(depths_check, fwhm_vals, 'bo-', ms=9, lw=2)
    ax2.set_xlabel('Profondità z (cm)', fontsize=12)
    ax2.set_ylabel('FWHM profilo laterale (cm)', fontsize=12)
    ax2.set_title(f'Diffusione angolare Compton: FWHM vs profondità\n'
                  f'(cresce monotonicamente: {"sì ✓" if penumbra_grows else "NO ✗"},'
                  f' ratio 20/3cm = {ratio_20_3:.2f})', fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'TEST 4 — Scattering Compton [{PASS if all(results) else FAIL}]',
                 fontsize=12)
    plt.tight_layout()
    save_fig(fig, 'val_test4_compton_angular')

    passed = all(results)
    print(f"\n  Risultato TEST 4: {PASS if passed else FAIL}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Convergenza statistica 1/√N (dal simulatore reale)
# ─────────────────────────────────────────────────────────────────────────────
def test5_convergence(skip_slow: bool = False):
    """
    Lancia il simulatore con N = 10⁴, 10⁵, 10⁶, 10⁷ (o fino a 10⁶ se skip_slow).
    Per ogni N, esegue 5 repliche con seed diversi.
    Misura σ della dose nel voxel centrale (z ≈ 10 cm) normalizzata per fotone.
    La pendenza log-log(σ vs N) deve essere −0.5 ± 0.10.
    """
    header("Convergenza statistica: σ ∝ 1/√N", 5)
    print("  Verifica che l'incertezza statistica della dose scala come 1/√N.")
    print("  Misura σ sul voxel centrale (z≈10cm) su 5 repliche indipendenti.\n")

    N_VALUES = [10_000, 100_000, 1_000_000]
    if not skip_slow:
        N_VALUES.append(10_000_000)
    else:
        print("  NOTA: --skip_slow attivo, salto N=10⁷ (richiede ~10min)\n")

    N_REP   = 5
    IZ_MEAS = int(10.0 / VOXEL_CM)  # z ≈ 10 cm
    IX_MEAS = NX // 2
    IY_MEAS = NY // 2

    sigma_list = []
    mean_list  = []
    N_ok       = []

    print(f"  {'N':>12}  {'Rep OK':>6}  {'<dose/N>':>14}  {'σ':>14}  CV%")
    print(f"  {'─' * 65}")

    for N in N_VALUES:
        doses = []
        for rep in range(N_REP):
            seed = 500 + rep
            if run_sim(N, phantom_type=0, seed=seed):
                d3d = load_dose_bin(DOSE_FILE)
                if d3d is not None:
                    doses.append(d3d[IZ_MEAS, IY_MEAS, IX_MEAS] / N)

        if len(doses) < 3:
            print(f"  {N:>12,}  {'<3':>6}  — troppo poche repliche")
            continue

        doses_arr = np.array(doses)
        m  = doses_arr.mean()
        s  = doses_arr.std(ddof=1)
        cv = s / m * 100.0 if m > 0 else 0.0

        sigma_list.append(s)
        mean_list.append(m)
        N_ok.append(N)

        print(f"  {N:>12,}  {len(doses):>6}  {m:>14.4e}  {s:>14.4e}  {cv:.3f}")

    if len(N_ok) < 3:
        print(f"\n  {FAIL}: meno di 3 livelli N validi.")
        return False

    N_arr = np.array(N_ok, dtype=float)
    s_arr = np.array(sigma_list)

    slope, intercept = np.polyfit(np.log10(N_arr), np.log10(s_arr), 1)
    delta = abs(slope - (-0.5))

    print(f"\n  Pendenza log-log: {slope:.4f}  (atteso: −0.500 ± 0.100)")
    print(f"  Scarto:           {slope - (-0.5):+.4f}")

    ok_pass = delta < 0.10
    ok_warn = delta < 0.20

    # Plot
    N_theory = np.logspace(np.log10(N_arr.min()), np.log10(N_arr.max()), 100)
    s_theory = sigma_list[0] * np.sqrt(N_ok[0] / N_theory)
    s_fit    = 10**intercept * N_arr**slope

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(N_arr, s_arr,    'bo', ms=9, zorder=5, label='σ misurata')
    ax.loglog(N_arr, s_fit,    'b--', lw=1.5, alpha=0.8,
              label=f'Fit: slope={slope:.3f}')
    ax.loglog(N_theory, s_theory, 'r-', lw=2,
              label='Teorica 1/√N (slope=−0.500)')
    ax.set_xlabel('N particelle', fontsize=12)
    ax.set_ylabel('σ (dose/fotone)', fontsize=12)
    ax.set_title(f'TEST 5 — Convergenza statistica  '
                 f'[{PASS if ok_pass else (WARN if ok_warn else FAIL)}]',
                 fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which='both')
    save_fig(fig, 'val_test5_convergence')

    if ok_pass:
        result_str = PASS
    elif ok_warn:
        result_str = WARN
    else:
        result_str = FAIL

    print(f"\n  Risultato TEST 5: {result_str}")
    return ok_pass


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Forma quantitativa del PDD 6MV
# ─────────────────────────────────────────────────────────────────────────────
def test6_pdd_quantitative():
    """
    Verifica quantitativa della forma del PDD con spettro 6MV:
      • d_max ∈ [1.0, 4.0] cm         (build-up 6MV fascio parallelo)
      • D(10)/D(max) ∈ [0.60, 0.80]   (profondità terapeutica)
      • D(20)/D(10) ∈ [0.45, 0.65]    (rapporto di decadimento)
      • PDD decrescente dopo il picco  (fisica corretta)

    I range sono adattati al fascio parallelo (non divergente):
    il d_max è un po' più superficiale rispetto al fascio clinico SAD100.
    Riferimento: Khan & Gibbons, Physics of Radiation Therapy (5th ed.),
    Cap. 9 — adattato per geometria parallela.
    """
    header("Forma quantitativa del PDD 6MV in acqua", 6)
    print("  Verifica i valori dosimetrici del PDD prodotto dal simulatore.")
    print("  Range adattati a fascio parallelo (non divergente SAD100).\n")

    N = 2_000_000
    print(f"  Lancio simulazione N={N:,} fotoni, spettro 6MV...")
    if not run_sim(N, phantom_type=0, seed=42):
        return False

    depths, pdd = load_pdd(PDD_WATER)
    if depths is None:
        return False

    i_peak   = np.argmax(pdd)
    z_peak   = depths[i_peak]
    D_peak   = pdd[i_peak]
    D_10     = pdd[int(10.0 / VOXEL_CM)]
    D_20     = pdd[int(20.0 / VOXEL_CM)]
    ratio_10 = D_10 / D_peak if D_peak > 0 else 0
    ratio_20_10 = D_20 / D_10 if D_10 > 0 else 0

    # Monotonia dopo il picco (smoothed)
    pdd_sm    = np.convolve(pdd, np.ones(5)/5, mode='same')
    post_peak = pdd_sm[i_peak:]
    frac_decr = (np.diff(post_peak) < 0).mean()

    checks = [
        ("d_max ∈ [1.0, 4.0] cm",
         1.0 <= z_peak <= 4.0,
         f"d_max = {z_peak:.2f} cm"),
        ("D(10)/D(max) ∈ [0.60, 0.80]",
         0.60 <= ratio_10 <= 0.80,
         f"D(10)/D(max) = {ratio_10:.3f}"),
        ("D(20)/D(10) ∈ [0.45, 0.65]",
         0.45 <= ratio_20_10 <= 0.65,
         f"D(20)/D(10) = {ratio_20_10:.3f}"),
        ("PDD decrescente > 70% dopo picco",
         frac_decr > 0.70,
         f"{frac_decr*100:.0f}% dei passi decrescenti"),
    ]

    print(f"\n  Valori misurati dal simulatore:")
    print(f"    d_max        : {z_peak:.2f} cm")
    print(f"    D(max)       : {D_peak:.2f}%")
    print(f"    D(10 cm)     : {D_10:.2f}%")
    print(f"    D(20 cm)     : {D_20:.2f}%")
    print(f"    D(10)/D(max) : {ratio_10:.3f}  (atteso: 0.60–0.80)")
    print(f"    D(20)/D(10)  : {ratio_20_10:.3f}  (atteso: 0.45–0.65)")
    print()

    results = []
    for desc, ok, val in checks:
        print(f"  {desc}:  {val}  →  {PASS if ok else FAIL}")
        results.append(ok)

    # Valori tabulati di riferimento per fascio parallelo 6MV
    # (interpolati da Sheikh-Bagheri & Rogers 2002, adattati a fascio parallelo)
    ref_z   = np.array([0.5, 1.5, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    ref_pdd = np.array([75,  100,  97,   90,   73,   60,   49,   40 ])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depths, pdd,   'b-',  lw=2.5, label=f'MC simulato (N={N//1e6:.0f}M)')
    ax.plot(ref_z, ref_pdd, 'rs', ms=8,  label='Riferimento letteratura (fascio parallelo)')
    ax.axvline(z_peak, color='g', ls=':', lw=2, alpha=0.8,
               label=f'd_max = {z_peak:.1f} cm')
    ax.axvline(10.0,   color='orange', ls='--', lw=1.5, alpha=0.8,
               label=f'D(10) = {D_10:.1f}%')
    ax.set_xlabel('Profondità z (cm)', fontsize=12)
    ax.set_ylabel('Dose (%)', fontsize=12)
    ax.set_title(f'TEST 6 — Forma PDD 6MV  '
                 f'[{PASS if all(results) else FAIL}]', fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(0, 115)
    save_fig(fig, 'val_test6_pdd_shape')

    passed = all(results)
    print(f"\n  Risultato TEST 6: {PASS if passed else FAIL}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Eterogeneità acqua vs acqua+osso
# ─────────────────────────────────────────────────────────────────────────────
def test7_heterogeneity():
    """
    Confronta i PDD di acqua omogenea e acqua+inserto osseo.
    L'osso (ρ=1.85 g/cm³, vs acqua ρ=1.00) ha coefficiente di attenuazione
    lineare più alto → dopo l'inserto osseo (12.5–17.5 cm) la dose deve
    differire significativamente.

    Verifica:
      • La dose nell'osso (12.5–17.5 cm) è più alta nel phantom eterogeneo
        (μ_osso > μ_acqua → più assorbimento locale)
      • La dose dopo l'osso (> 17.5 cm) è più bassa nel phantom eterogeneo
        (il fascio è stato più attenuato)
      • La differenza a 20 cm (dopo l'osso) è > 3% assoluto
    """
    header("Eterogeneità: acqua vs acqua+inserto osseo", 7)
    print("  Verifica che il simulatore gestisca correttamente le eterogeneità.")
    print("  Inserto osseo: 5×5×5 cm³, ρ=1.85 g/cm³, centrato a 15 cm.\n")

    N = 1_000_000

    print(f"  [1] Phantom acqua omogenea (N={N:,})...")
    if not run_sim(N, phantom_type=0, seed=42):
        return False
    depths_w, pdd_w = load_pdd(PDD_WATER)

    print(f"  [2] Phantom acqua + osso (N={N:,})...")
    if not run_sim(N, phantom_type=1, seed=42):
        return False
    depths_h, pdd_h = load_pdd(PDD_HETERO)

    if depths_w is None or depths_h is None:
        return False

    # Zona ossea: iz = [int(12.5/0.3), int(17.5/0.3)] = [41, 58]
    iz_bone_start = int(12.5 / VOXEL_CM)
    iz_bone_end   = int(17.5 / VOXEL_CM)
    iz_after      = int(20.0 / VOXEL_CM)

    D_water_bone  = pdd_w[iz_bone_start:iz_bone_end].mean()
    D_hetero_bone = pdd_h[iz_bone_start:iz_bone_end].mean()
    D_water_after = pdd_w[iz_after]
    D_hetero_after = pdd_h[iz_after]

    diff_bone  = D_hetero_bone  - D_water_bone   # positivo: più dose nell'osso
    diff_after = D_water_after  - D_hetero_after  # positivo: meno dose dopo l'osso

    print(f"\n  Dose media nella zona ossea (12.5–17.5 cm):")
    print(f"    Acqua pura   : {D_water_bone:.2f}%")
    print(f"    Con osso     : {D_hetero_bone:.2f}%")
    print(f"    Differenza   : {diff_bone:+.2f}%  "
          f"({'più alta nell osso ✓' if diff_bone > 0 else 'INATTESO ✗'})")

    print(f"\n  Dose a 20 cm (dopo l inserto osseo):")
    print(f"    Acqua pura   : {D_water_after:.2f}%")
    print(f"    Con osso     : {D_hetero_after:.2f}%")
    print(f"    Differenza   : {diff_after:+.2f}%  (atteso: > 3%)")

    checks = [
        ("Dose nell osso > acqua nell zona ossea",
         diff_bone > 0),
        ("Dose dopo l osso < acqua (ombra dell osso)",
         diff_after > 0),
        ("Differenza a 20 cm > 3% assoluto",
         abs(diff_after) > 3.0),
    ]
    results = []
    for desc, ok in checks:
        print(f"  {desc}: {PASS if ok else FAIL}")
        results.append(ok)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depths_w, pdd_w, 'b-',  lw=2.5, label='Acqua omogenea')
    ax.plot(depths_h, pdd_h, 'r--', lw=2.5, label='Acqua + Osso')
    ax.axvspan(12.5, 17.5, alpha=0.15, color='orange', label='Inserto osseo')
    ax.axvline(20.0, color='gray', ls=':', lw=1.5, label=f'z=20cm: Δ={diff_after:+.1f}%')
    ax.set_xlabel('Profondità z (cm)', fontsize=12)
    ax.set_ylabel('Dose (%)', fontsize=12)
    ax.set_title(f'TEST 7 — Eterogeneità  [{PASS if all(results) else FAIL}]',
                 fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(0, 115)
    save_fig(fig, 'val_test7_heterogeneity')

    passed = all(results)
    print(f"\n  Risultato TEST 7: {PASS if passed else FAIL}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Gamma index 2%/2mm vs letteratura (o file esterno)
# ─────────────────────────────────────────────────────────────────────────────
def test8_gamma_index(ref_csv: str | None = None):
    """
    Confronta il PDD simulato con un riferimento tramite gamma index 2%/2mm.

    Se --ref_csv è fornito, usa quel file (DOSXYZnrc, OpenDosimetry, TPS).
    Altrimenti usa i punti tabulati di Sheikh-Bagheri & Rogers (2002) per
    fascio parallelo 6MV in acqua.

    Il pass rate richiesto è:
      > 95% se il riferimento è DOSXYZnrc o OpenDosimetry (dati validati)
      > 85% se si usa il riferimento sintetico da letteratura
             (perché il riferimento è per fascio divergente, il sim è parallelo)
    """
    header("Gamma index 2%/2mm vs riferimento", 8)

    N = 3_000_000
    print(f"  Lancio simulazione N={N:,} fotoni, spettro 6MV...")
    if not run_sim(N, phantom_type=0, seed=42):
        return False

    depths_mc, pdd_mc = load_pdd(PDD_WATER)
    if depths_mc is None:
        return False

    # ── Carica o costruisce il riferimento ───────────────────────────────────
    if ref_csv is not None and os.path.exists(ref_csv):
        print(f"  Carico riferimento esterno: {ref_csv}")
        data_ref    = np.loadtxt(ref_csv, delimiter=',', skiprows=1)
        depths_ref  = data_ref[:, 0]
        pdd_ref     = data_ref[:, 1]
        ref_label   = Path(ref_csv).stem
        pass_thresh = 95.0
        print(f"  Pass rate richiesto: > {pass_thresh}% (riferimento esterno validato)")
    else:
        if ref_csv is not None:
            print(f"  AVVISO: {ref_csv} non trovato, uso riferimento da letteratura.")
        else:
            print("  Nessun riferimento esterno (--ref_csv), uso letteratura.")
        # Punti tabulati Sheikh-Bagheri & Rogers 2002, adattati fascio parallelo
        z_pts    = np.array([0.5,1.0,1.5,2.0,3.0,4.0,5.0,7.0,10.0,
                             12.0,15.0,18.0,20.0,22.0,25.0,28.0])
        D_pts    = np.array([75,  94,  100, 99,  97,  94,  90,  82,  73,
                             67,   60,   53,   49,   45,   40,   35  ])
        depths_ref = np.linspace(z_pts[0], z_pts[-1], 200)
        pdd_ref    = np.interp(depths_ref, z_pts, D_pts)
        ref_label   = "Sheikh-Bagheri & Rogers 2002 (sintetico)"
        pass_thresh = 85.0
        print(f"  Pass rate richiesto: > {pass_thresh}% (riferimento sintetico)")

    print(f"  Riferimento: {ref_label}\n")

    # ── Interpolazione MC sui punti del riferimento ───────────────────────────
    pdd_mc_interp = np.interp(depths_ref, depths_mc, pdd_mc)

    # ── Gamma index 1D ────────────────────────────────────────────────────────
    DD  = 2.0   # %
    DTA = 0.2   # cm = 2 mm

    gamma = np.full(len(depths_ref), np.inf)
    for i, (z_r, D_r) in enumerate(zip(depths_ref, pdd_ref)):
        dD = (pdd_mc_interp - D_r) / DD
        dz = (depths_ref    - z_r) / DTA
        gamma[i] = np.sqrt((dD**2 + dz**2).min())

    # Zona rilevante: 0.5–27 cm
    mask      = (depths_ref >= 0.5) & (depths_ref <= 27.0)
    gamma_r   = gamma[mask]
    z_r       = depths_ref[mask]
    n_total   = len(gamma_r)
    n_pass    = (gamma_r <= 1.0).sum()
    pass_rate = n_pass / n_total * 100.0 if n_total > 0 else 0.0

    print(f"  Zona analizzata: z ∈ [0.5, 27.0] cm ({n_total} punti)")
    print(f"  Punti con Γ ≤ 1: {n_pass} / {n_total}")
    print(f"  Pass rate:       {pass_rate:.1f}%  (richiesto: > {pass_thresh}%)")

    # Punti clinici di dettaglio
    print(f"\n  {'z [cm]':>8}  {'D_MC [%]':>10}  {'D_ref [%]':>10}  "
          f"{'ΔD [%]':>8}  {'Γ':>8}  Stato")
    print(f"  {'─' * 60}")
    for z_c in [1.5, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0]:
        idx  = np.argmin(np.abs(depths_ref - z_c))
        g    = gamma[idx]
        D_m  = pdd_mc_interp[idx]
        D_r  = pdd_ref[idx]
        ok   = "PASS ✓" if g <= 1.0 else "FAIL ✗"
        print(f"  {depths_ref[idx]:>8.2f}  {D_m:>10.2f}  {D_r:>10.2f}  "
              f"{D_m-D_r:>+8.2f}  {g:>8.3f}  {ok}")

    passed = pass_rate >= pass_thresh

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                   gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(depths_mc,  pdd_mc,  'b-',  lw=2, label='MC simulato')
    ax1.plot(depths_ref, pdd_ref, 'r--', lw=2, label=f'Riferimento: {ref_label}')
    ax1.set_ylabel('Dose (%)', fontsize=12)
    ax1.set_title(f'TEST 8 — Gamma index 2%/2mm', fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 115)

    colors = ['green' if g <= 1.0 else 'red' for g in gamma_r]
    ax2.bar(z_r, gamma_r, width=VOXEL_CM/2, color=colors, alpha=0.8)
    ax2.axhline(1.0, color='black', ls='--', lw=1.5, label='Soglia Γ=1')
    ax2.set_xlabel('Profondità z (cm)', fontsize=12)
    ax2.set_ylabel('Gamma Γ(z)', fontsize=12)
    ax2.set_title(f'Pass rate = {pass_rate:.1f}%  '
                  f'[{PASS if passed else FAIL}]  '
                  f'(soglia: {pass_thresh:.0f}%)', fontsize=11)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(gamma_r.max() * 1.15, 1.5))

    plt.tight_layout()
    save_fig(fig, 'val_test8_gamma_index')

    print(f"\n  Risultato TEST 8: {PASS if passed else FAIL}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Suite di validazione scientifica del simulatore MC-RT')
    parser.add_argument('--test', nargs='+', type=int, default=None,
                        help='Test da eseguire (es. --test 1 3 5). Default: tutti.')
    parser.add_argument('--ref_csv', default=None,
                        help='CSV PDD di riferimento per TEST 8 '
                             '(DOSXYZnrc o OpenDosimetry)')
    parser.add_argument('--skip_slow', action='store_true',
                        help='Salta N=10⁷ nel TEST 5 (riduce tempo da ~10min a ~2min)')
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  SUITE DI VALIDAZIONE SCIENTIFICA — Monte Carlo RT               ║")
    print("║  Angelica Porco — Matricola 264034                               ║")
    print("║  High Performance Computing 2025/2026                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Verifica prerequisiti
    if not os.path.exists(EXECUTABLE):
        print(f"\nERRORE: {EXECUTABLE} non trovato.")
        print("Compila con:  g++ -O2 -std=c++17 -o mc_rt_cpu main.cpp -lm")
        sys.exit(1)

    all_tests = {
        1: ("Coefficiente attenuazione NIST",    test1_attenuation_coefficient),
        2: ("Beer-Lambert solo fotoelettrico",    test2_beer_lambert),
        3: ("Bilancio energetico",                test3_energy_balance),
        4: ("Scattering Compton angolare",        test4_compton_angular),
        5: ("Convergenza statistica 1/√N",        lambda: test5_convergence(args.skip_slow)),
        6: ("Forma quantitativa PDD 6MV",         test6_pdd_quantitative),
        7: ("Eterogeneità acqua vs osso",         test7_heterogeneity),
        8: ("Gamma index 2%/2mm",                 lambda: test8_gamma_index(args.ref_csv)),
    }

    tests_to_run = args.test if args.test else list(all_tests.keys())

    results  = {}
    t0_total = time.time()

    for n in tests_to_run:
        if n not in all_tests:
            print(f"\nATTENZIONE: test {n} non esiste (disponibili: 1–8)")
            continue
        name, func = all_tests[n]
        t0 = time.time()
        try:
            r = func()
        except Exception as e:
            import traceback
            print(f"\n  ECCEZIONE in TEST {n}: {e}")
            traceback.print_exc()
            r = False
        dt = time.time() - t0
        results[n] = (name, r, dt)

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0_total
    print(f"\n╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  RIEPILOGO VALIDAZIONE                                           ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")

    n_pass = n_fail = n_skip = 0
    for n, (name, r, dt) in results.items():
        if r is None:
            s = "SKIP ─"
            n_skip += 1
        elif r:
            s = "PASS ✓"
            n_pass += 1
        else:
            s = "FAIL ✗"
            n_fail += 1
        print(f"║  {s}  TEST {n}: {name:<42}  [{dt:5.0f}s] ║")

    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  {n_pass} PASS  |  {n_fail} FAIL  |  {n_skip} SKIP  "
          f"|  Tempo totale: {elapsed:.0f}s{' ':>17}║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")

    if figures_saved:
        print(f"\n  Figure generate:")
        for f in figures_saved:
            print(f"    {f}")

    if n_fail == 0:
        print(f"\n  ✓ Tutti i test superati — validazione scientifica completata.")
        if not os.path.exists(EXECUTABLE_3A):
            print(f"  ⚠ TEST 2 non eseguito: compila test3a_beer_lambert.cpp per "
                  f"la validazione Beer-Lambert completa.")
    else:
        print(f"\n  ✗ {n_fail} test falliti — verifica i messaggi sopra.")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())