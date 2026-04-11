#!/usr/bin/env python3
"""
test3c_gamma_index.py
─────────────────────────────────────────────────────────────────────────────
TEST 3c — Confronto con DOSXYZnrc tramite gamma index 2%/2mm

NOTA IMPORTANTE SU DOSXYZnrc SU COLAB:
  L'istruzione originale (!git clone EGSnrc + pip install) non funziona
  perché EGSnrc richiede una compilazione Fortran complessa (make + mortran).
  Su Colab è necessario usare una distribuzione pre-compilata o un container.

  Questo script gestisce due modalità:
    A) MODALITÀ REALE:   DOSXYZnrc disponibile → carica il suo output
    B) MODALITÀ FALLBACK: DOSXYZnrc non disponibile → usa OpenDosimetry
       (database pubblico di PDD validati per fascio 6MV Varian, validato
       contro multiple istituzioni, DOI: 10.1002/mp.13448)

  Per l'esame, il confronto di riferimento più pratico è OpenDosimetry:
  scarica il CSV del PDD 6MV dal sito, passalo con --ref_csv.

Uso:
  # Con il file di riferimento da DOSXYZnrc (o OpenDosimetry):
  python3 test3c_gamma_index.py --mc_csv pdd_water.csv --ref_csv ref_pdd.csv

  # Con file binario dose (più preciso, usa tutta la griglia 3D):
  python3 test3c_gamma_index.py --mc_bin dose_water.bin --ref_csv ref_pdd.csv

  # Solo gamma index tra due CSV qualsiasi (generico):
  python3 test3c_gamma_index.py --mc_csv pdd_1.csv --ref_csv pdd_2.csv

Formato CSV atteso (MC e riferimento):
  depth_cm,dose_percent
  0.15,82.3
  0.45,90.1
  ...

Criteri gamma index:
  ΔD   = 2%  (percentuale della dose massima)
  Δd   = 2mm = 0.2 cm
  Pass rate richiesto: > 95%

Output:
  - Tabella stdout: gamma index per ogni punto
  - File PNG: test3c_gamma_map.png
  - File CSV: test3c_gamma_results.csv
  - Stampa pass rate e risultato PASS/FAIL

Istruzioni per generare il file DOSXYZnrc di riferimento:
  1. Scarica EGSnrc pre-compilato:
       https://github.com/nrc-cnrc/EGSnrc/releases
  2. Crea il file .egsinp per DOSXYZnrc con:
       - Phantom 100x100x100 voxel, voxel 0.3cm
       - Materiale WATER_LIQUID
       - Sorgente: fascio parallelo 10x10cm², spettro 6MV (file .spectrum)
       - ECUT=0.521 MeV, PCUT=0.010 MeV
       - Nstory = 10^7
  3. Esegui DOSXYZnrc e converti l'output .3ddose in CSV
       (usa lo script dosxyz_to_csv.py nella cartella EGSnrc/scripts/)

Alternativa consigliata — OpenDosimetry:
  Scarica il PDD di riferimento da:
  https://app.opendosimetry.org → Photon → 6MV → 10x10 → PDD
  Il file è già in formato CSV con colonne depth_mm e dose_percent.
  Converti la colonna depth_mm → depth_cm prima di passarlo a --ref_csv.

Autore: Angelica Porco — Matricola 264034
Corso:  High Performance Computing 2025/2026
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────
NX = NY = NZ = 100
VOXEL_CM = 0.30

GAMMA_DD    = 2.0   # % dose difference
GAMMA_DTA   = 0.2   # cm = 2 mm distance-to-agreement
PASS_THRESH = 95.0  # % pass rate richiesto


# ─────────────────────────────────────────────────────────────────────────────
# GAMMA INDEX 1D (lungo asse z)
#
# Per ogni punto r nel PDD di riferimento:
#   Γ(r) = min_{r'} sqrt( [(D_MC(r') - D_ref(r)) / (ΔD)]²
#                       + [(r' - r) / (Δd)]² )
#
# dove D è in % (normalizzata al max), r in cm.
# Il punto supera il test se Γ(r) ≤ 1.
# ─────────────────────────────────────────────────────────────────────────────
def gamma_index_1d(z_ref: np.ndarray, D_ref: np.ndarray,
                   z_mc: np.ndarray,  D_mc: np.ndarray,
                   dd_percent: float = GAMMA_DD,
                   dta_cm:     float = GAMMA_DTA) -> np.ndarray:
    """
    Calcola il gamma index 1D punto per punto.

    Parametri
    ---------
    z_ref, D_ref : array di profondità [cm] e dose [%] del riferimento
    z_mc,  D_mc  : idem per la distribuzione MC da testare
    dd_percent   : tolleranza dose [%]
    dta_cm       : tolleranza distanza [cm]

    Restituisce
    -----------
    gamma : array della stessa lunghezza di z_ref, con Γ(r) per ogni punto
    """
    gamma = np.full(len(z_ref), np.inf)

    for i, (z_r, D_r) in enumerate(zip(z_ref, D_ref)):
        # Per ogni punto del riferimento, cerca il minimo Γ su tutti i punti MC
        delta_D = (D_mc - D_r) / dd_percent
        delta_z = (z_mc - z_r) / dta_cm
        gamma_sq = delta_D**2 + delta_z**2
        gamma[i] = np.sqrt(gamma_sq.min())

    return gamma


def load_pdd_csv(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica un CSV PDD con colonne depth_cm, dose_percent.
    Restituisce (depth_cm, dose_percent).
    """
    if not os.path.exists(filename):
        print(f"ERRORE: file non trovato: {filename}")
        sys.exit(1)
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]


def load_pdd_from_bin(bin_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica dose_water.bin e ricava il PDD asse-z (media su finestra 8 voxel
    attorno all'asse centrale, coerente con compute_pdd() di output.h).
    """
    if not os.path.exists(bin_file):
        print(f"ERRORE: file non trovato: {bin_file}")
        sys.exit(1)

    dose_flat = np.fromfile(bin_file, dtype=np.float64)
    if dose_flat.size != NX * NY * NZ:
        print(f"ERRORE: dimensione file errata ({dose_flat.size} vs {NX*NY*NZ})")
        sys.exit(1)

    # Ricostruisce: dose_flat[ix + NX*(iy + NY*iz)] → dose_3d[iz, iy, ix]
    dose_3d = dose_flat.reshape((NZ, NY, NX))

    cx, cy    = NX // 2, NY // 2
    avg_half  = 8   # ±8 voxel = ±2.4 cm (come output.h)

    pdd_raw = np.zeros(NZ)
    for iz in range(NZ):
        region = dose_3d[iz,
                         max(0, cy-avg_half) : min(NY, cy+avg_half+1),
                         max(0, cx-avg_half) : min(NX, cx+avg_half+1)]
        pdd_raw[iz] = region.mean()

    max_d = pdd_raw.max()
    pdd   = pdd_raw / max_d * 100.0 if max_d > 0 else pdd_raw
    depths = (np.arange(NZ) + 0.5) * VOXEL_CM
    return depths, pdd


def make_synthetic_reference() -> tuple[np.ndarray, np.ndarray]:
    """
    Genera un PDD di riferimento sintetico per fascio 6MV parallelo in acqua,
    basato su valori tabulati da letteratura (Sheikh-Bagheri & Rogers 2002,
    Khan & Gibbons 2014 — adattato a fascio parallelo).

    Usato SOLO se non viene fornito un file di riferimento esterno.
    NON usare per la relazione finale: usare DOSXYZnrc o OpenDosimetry.
    """
    # Punti di controllo da letteratura (fascio parallelo, non divergente)
    z_pts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                      7.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0])
    D_pts = np.array([75.0, 88.0, 97.0, 100.0, 99.5, 97.0, 94.0, 90.0,
                      82.0, 73.0, 67.0, 60.0, 53.0, 49.0, 45.0, 40.0, 35.0, 32.0])

    z_fine = np.arange(0.5*VOXEL_CM, NZ*VOXEL_CM, VOXEL_CM/2)
    D_fine = np.interp(z_fine, z_pts, D_pts)
    return z_fine, D_fine


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="TEST 3c — Confronto MC vs riferimento con gamma index 2%%/2mm")
    parser.add_argument('--mc_csv',  default='pdd_water.csv',
                        help='CSV PDD del simulatore MC (default: pdd_water.csv)')
    parser.add_argument('--mc_bin',  default=None,
                        help='Binario dose_water.bin (alternativa a --mc_csv)')
    parser.add_argument('--ref_csv', default=None,
                        help='CSV PDD di riferimento (DOSXYZnrc o OpenDosimetry)')
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  TEST 3c — Gamma index 2%%/2mm vs riferimento               ║")
    print("║  Angelica Porco — Matricola 264034                           ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── 1. Carica il PDD del simulatore MC ───────────────────────────────────
    if args.mc_bin is not None:
        print(f"  Carico PDD MC da file binario: {args.mc_bin}")
        z_mc, D_mc = load_pdd_from_bin(args.mc_bin)
    else:
        print(f"  Carico PDD MC da CSV: {args.mc_csv}")
        z_mc, D_mc = load_pdd_csv(args.mc_csv)

    print(f"  Punti PDD MC: {len(z_mc)},  range z: [{z_mc[0]:.2f}, {z_mc[-1]:.2f}] cm")

    # ── 2. Carica il PDD di riferimento ──────────────────────────────────────
    if args.ref_csv is not None:
        print(f"  Carico PDD di riferimento da: {args.ref_csv}")
        z_ref, D_ref = load_pdd_csv(args.ref_csv)
        ref_source   = args.ref_csv
    else:
        print("  AVVISO: nessun file di riferimento fornito (--ref_csv).")
        print("  Uso PDD sintetico da letteratura (SOLO per test preliminare).")
        print("  Per la relazione finale usare DOSXYZnrc o OpenDosimetry.\n")
        z_ref, D_ref = make_synthetic_reference()
        ref_source   = "letteratura (sintetico)"

    print(f"  Punti PDD riferimento: {len(z_ref)},  "
          f"range z: [{z_ref[0]:.2f}, {z_ref[-1]:.2f}] cm\n")

    # ── 3. Interpolazione del PDD MC sui punti del riferimento ────────────────
    # Per il gamma index è più preciso interpolare MC sui punti ref che viceversa
    D_mc_interp = np.interp(z_ref, z_mc, D_mc)

    # ── 4. Calcolo gamma index ────────────────────────────────────────────────
    print(f"  Calcolo gamma index (ΔD={GAMMA_DD}%, Δd={GAMMA_DTA*10:.0f}mm)...")
    gamma = gamma_index_1d(z_ref, D_ref, z_mc, D_mc,
                           dd_percent=GAMMA_DD, dta_cm=GAMMA_DTA)

    # Considera solo la zona dosimetricamente rilevante: z ∈ [0.5, 27] cm
    # (esclude superficie e coda con poca statistica)
    z_mask   = (z_ref >= 0.5) & (z_ref <= 27.0)
    gamma_r  = gamma[z_mask]
    z_r      = z_ref[z_mask]
    D_ref_r  = D_ref[z_mask]
    D_mc_r   = D_mc_interp[z_mask]

    n_total  = len(gamma_r)
    n_pass   = (gamma_r <= 1.0).sum()
    pass_rate = n_pass / n_total * 100.0 if n_total > 0 else 0.0

    # ── 5. Tabella dettagliata ai punti di riferimento clinici ────────────────
    print(f"\n  Tabella gamma index ai punti clinici:\n")
    print(f"  {'z [cm]':>8}  {'D_ref [%]':>10}  {'D_MC [%]':>10}  "
          f"{'ΔD [%]':>8}  {'Γ':>8}  Stato")
    print(f"  {'─' * 60}")

    z_clinical = [1.5, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    for z_c in z_clinical:
        idx = np.argmin(np.abs(z_ref - z_c))
        g   = gamma[idx]
        dD  = D_mc_interp[idx] - D_ref[idx]
        ok  = "PASS ✓" if g <= 1.0 else "FAIL ✗"
        print(f"  {z_ref[idx]:>8.2f}  {D_ref[idx]:>10.2f}  {D_mc_interp[idx]:>10.2f}  "
              f"{dD:>+8.2f}  {g:>8.3f}  {ok}")

    # ── 6. Risultato ──────────────────────────────────────────────────────────
    passed = pass_rate >= PASS_THRESH

    print(f"\n  ── Risultato gamma index ─────────────────────────────────")
    print(f"  Zona analizzata   : z ∈ [0.5, 27.0] cm  ({n_total} punti)")
    print(f"  Punti con Γ ≤ 1   : {n_pass} / {n_total}")
    print(f"  Pass rate         : {pass_rate:.1f}%  (richiesto: > {PASS_THRESH:.0f}%)")
    print(f"  Riferimento usato : {ref_source}")
    print(f"\n  ══════════════════════════════════════════════════════════")
    print(f"  Risultato TEST 3c: {'PASS ✓' if passed else 'FAIL ✗'}")
    if not passed:
        print(f"\n  Possibili cause del fallimento:")
        print(f"    • Il riferimento sintetico non corrisponde esattamente al codice")
        print(f"    • Statistica insufficiente nel PDD MC (aumentare N a 10⁷)")
        print(f"    • Differenze nel setup (cutoff, dimensioni campo, spettro)")
    print(f"  ══════════════════════════════════════════════════════════")

    # ── 7. Grafici ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Pannello superiore: PDD
    ax1.plot(z_mc,  D_mc,          'b-',  lw=2, label='MC simulato')
    ax1.plot(z_ref, D_ref,         'r--', lw=2, label=f'Riferimento ({ref_source})')
    ax1.set_ylabel('Dose (%)',  fontsize=12)
    ax1.set_title(f'TEST 3c — Confronto PDD  |  gamma index 2%/2mm', fontsize=12)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 110)

    # Pannello inferiore: gamma index
    colors = ['green' if g <= 1.0 else 'red' for g in gamma_r]
    ax2.bar(z_r, gamma_r, width=VOXEL_CM, color=colors, alpha=0.7, label='Γ(z)')
    ax2.axhline(1.0, color='black', ls='--', lw=1.5, label='Soglia Γ = 1')
    ax2.set_xlabel('Profondità z (cm)', fontsize=12)
    ax2.set_ylabel('Gamma Γ(z)', fontsize=12)
    ax2.set_title(f'Pass rate = {pass_rate:.1f}%  '
                  f'({"PASS ✓" if passed else "FAIL ✗"})', fontsize=11)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(gamma_r.max() * 1.15, 1.5))

    plt.tight_layout()
    fig.savefig('test3c_gamma_map.png', dpi=150)
    print(f"\n  Grafico salvato: test3c_gamma_map.png")

    # ── 8. Salva CSV ──────────────────────────────────────────────────────────
    with open('test3c_gamma_results.csv', 'w') as f:
        f.write("depth_cm,D_ref_percent,D_MC_percent,delta_D_percent,"
                "gamma,pass\n")
        for i, z_i in enumerate(z_r):
            f.write(f"{z_i:.3f},{D_ref_r[i]:.3f},{D_mc_r[i]:.3f},"
                    f"{D_mc_r[i]-D_ref_r[i]:+.3f},"
                    f"{gamma_r[i]:.4f},{1 if gamma_r[i]<=1 else 0}\n")
    print(f"  Dati salvati: test3c_gamma_results.csv")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
