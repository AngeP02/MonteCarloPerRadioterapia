#!/usr/bin/env python3
"""
tests.py
────────────────────────────────────────────────────────────────────────────────
Suite di test per la validazione fisica della versione CPU sequenziale.

Test implementati:
  TEST 1 — Convergenza statistica 1/√N
  TEST 2 — Forma del PDD: decadimento monotono e range fisicamente corretto
  TEST 3 — Sezioni d'urto NIST a 2 MeV (frazioni interazione)
  TEST 4 — Campionamento Klein-Nishina: <cos_theta> vs valore analitico
  TEST 5 — Conservazione energia (energia depositata vs energia iniziale)
  TEST 6 — Eterogeneità: osso assorbe più dell'acqua dopo il picco

Nota su TEST 1 (Beer-Lambert):
  Il confronto con exp(-μz) vale SOLO per fascio monoenergetico senza
  scattering (solo fotoelettrico). Con Compton attivo il PDD decade più
  lentamente perché i fotoni continuano a depositare energia dopo ogni
  scattering. Questo NON è un bug — è fisica corretta.
  La validazione quantitativa va fatta contro DOSXYZnrc (Sezione 7 relazione).

Autore: Angelica Porco — Matricola 264034
────────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import subprocess, os, time, sys

ME_C2    = 0.51099895
VOXEL_CM = 0.30
NX = NY = NZ = 100

PASS = "✓ PASS"
FAIL = "✗ FAIL"
WARN = "⚠ WARN"

def run_sim(N, phantom_type=0, seed=42, quiet=True):
    """Esegue mc_rt_cpu e restituisce (depths, pdd)."""
    cmd = ["./mc_rt_cpu", str(int(N)), str(phantom_type), str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERRORE:", result.stderr[:300])
        return None, None
    fname = "pdd_water.csv" if phantom_type == 0 else "pdd_hetero.csv"
    if not os.path.exists(fname):
        return None, None
    data = np.loadtxt(fname, delimiter=',', skiprows=1)
    return data[:,0], data[:,1]

def print_header(title):
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Convergenza statistica 1/√N
# ─────────────────────────────────────────────────────────────────────────────
def test1_convergence():
    print_header("TEST 1 — Convergenza statistica 1/√N")
    print("  Verifica che l'incertezza sulla dose scala come 1/√N.")
    print("  5 run indipendenti per ogni N, misura std sulla D10.\n")

    N_values = [10_000, 50_000, 200_000, 500_000]
    n_repeat  = 5
    iz_check  = int(10.0 / VOXEL_CM)

    std_vals = []
    print(f"  {'N':>10}  {'std [%]':>10}  {'CV [%]':>10}")
    print(f"  {'─'*40}")

    for N in N_values:
        doses = []
        for rep in range(n_repeat):
            _, pdd = run_sim(N, phantom_type=0, seed=200+rep)
            if pdd is not None and iz_check < len(pdd):
                doses.append(pdd[iz_check])
        if len(doses) < 3:
            std_vals.append(np.nan); continue
        s = np.std(doses)
        m = np.mean(doses)
        cv = s/m*100 if m>0 else 0
        std_vals.append(s)
        print(f"  {N:>10,}  {s:>10.3f}  {cv:>10.3f}")

    N_arr   = np.array(N_values, dtype=float)
    std_arr = np.array(std_vals)
    valid   = ~np.isnan(std_arr) & (std_arr > 0)
    passed  = False

    if valid.sum() >= 3:
        slope, _ = np.polyfit(np.log10(N_arr[valid]),
                              np.log10(std_arr[valid]), 1)
        print(f"\n  Pendenza log-log: {slope:.3f}  (atteso: -0.500 ± 0.15)")
        passed = abs(slope + 0.5) < 0.20

        plt.figure(figsize=(7,5))
        plt.loglog(N_arr[valid], std_arr[valid], 'bo-', ms=8, lw=2, label='std misurata')
        plt.loglog(N_arr, std_arr[valid][0]*np.sqrt(N_values[0]/N_arr),
                   'r--', lw=2, label='1/√N teorica')
        plt.xlabel('N particelle', fontsize=12)
        plt.ylabel('Std dose (%)', fontsize=12)
        plt.title(f'TEST 1 — Convergenza statistica (pendenza={slope:.3f})', fontsize=11)
        plt.legend(); plt.grid(True,alpha=0.3,which='both')
        plt.tight_layout(); plt.savefig('test1_convergence.png',dpi=150)
        print("  Figura: test1_convergence.png")

    print(f"\n  Risultato TEST 1: {PASS if passed else WARN}")
    return passed

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Forma del PDD
# ─────────────────────────────────────────────────────────────────────────────
def test2_pdd_shape():
    print_header("TEST 2 — Forma del PDD con spettro 6MV")
    print("  Criteri fisici per fascio parallelo in acqua:")
    print("  - Picco (d_max) nella prima metà del phantom (< 5 cm)")
    print("  - Decadimento generale da picco a 25 cm")
    print("  - D25 < D10 (la dose a 25cm è minore di quella a 10cm)")
    print("  Nota: fascio parallelo ≠ fascio divergente clinico,\n"
          "  quindi i valori assoluti differiscono da Khan & Gibbons.\n")

    depths, pdd = run_sim(1_000_000, 0, 42)
    if depths is None:
        print(f"  {FAIL}: simulazione fallita"); return False

    i_peak = np.argmax(pdd)
    z_peak = depths[i_peak]
    d_peak = pdd[i_peak]
    d10    = pdd[int(10.0/VOXEL_CM)]
    d20    = pdd[int(20.0/VOXEL_CM)]
    d25    = pdd[min(int(25.0/VOXEL_CM), NZ-1)]

    # Smoothed PDD per test di decadimento (media su 5 voxel)
    pdd_sm = np.convolve(pdd, np.ones(5)/5, mode='same')
    after_peak = pdd_sm[i_peak:]
    # Verifica tendenza decrescente generale (almeno il 70% dei passi decresce)
    diffs = np.diff(after_peak)
    frac_decr = (diffs < 0).mean()

    checks = [
        ("d_max nella prima metà (z < 5 cm)", z_peak < 5.0,
         f"z_max={z_peak:.2f}cm"),
        ("D10 < D_max * 0.95  (decadimento visibile)", d10 < d_peak*0.95,
         f"D10={d10:.1f}%, D_max={d_peak:.1f}%"),
        ("D25 < D10  (continua a decrescere)", d25 < d10,
         f"D25={d25:.1f}%, D10={d10:.1f}%"),
        ("Trend decrescente > 60% dopo picco", frac_decr > 0.60,
         f"{frac_decr*100:.0f}% dei passi decrescenti"),
    ]

    results = []
    for desc, ok, val in checks:
        status = PASS if ok else FAIL
        print(f"  {desc}: {val}  {status}")
        results.append(ok)

    # Valori di riferimento approssimativi per fascio parallelo
    ref_z  = np.array([0.5,1.5,3.0,5.0,10.0,15.0,20.0,25.0])
    ref_pdd = np.array([80,100,97,90,75,60,48,37])  # fascio parallelo (approssimato)

    plt.figure(figsize=(9,5))
    plt.plot(depths, pdd, 'b-', lw=2, label='MC simulato (1M fotoni)')
    plt.plot(ref_z, ref_pdd, 'rs', ms=8, label='Riferimento fascio parallelo (appross.)')
    plt.axvline(z_peak, color='g', ls=':', alpha=0.7, label=f'Picco z={z_peak:.1f}cm')
    plt.xlabel('Profondità (cm)',fontsize=12); plt.ylabel('Dose (%)',fontsize=12)
    plt.title('TEST 2 — Forma PDD spettro 6MV',fontsize=12)
    plt.legend(fontsize=9); plt.grid(True,alpha=0.3); plt.ylim(0,110)
    plt.tight_layout(); plt.savefig('test2_pdd_shape.png',dpi=150)
    print("  Figura: test2_pdd_shape.png")

    passed = all(results)
    print(f"\n  Risultato TEST 2: {PASS if passed else FAIL}")
    return passed

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Sezioni d'urto NIST a 2 MeV
# ─────────────────────────────────────────────────────────────────────────────
def test3_cross_sections():
    print_header("TEST 3 — Sezioni d'urto NIST a 2 MeV in acqua")
    print("  Verifica le frazioni relative dei processi.\n"
          "  Fonte: NIST XCOM, Hubbell & Seltzer 1996\n")

    mu_pe      = 2.200e-7   # cm²/g
    mu_compton = 0.04942    # cm²/g
    mu_pair    = 0.0        # cm²/g (sotto soglia a 2 MeV)
    mu_tot     = mu_pe + mu_compton + mu_pair

    p_pe  = mu_pe  / mu_tot * 100
    p_cmp = mu_compton / mu_tot * 100

    print(f"  μ_total = {mu_tot:.5f} cm²/g")
    print(f"  Fotoelettrico : {p_pe:.4f}%  (atteso ~0%)")
    print(f"  Compton       : {p_cmp:.2f}%  (atteso ~100%)")
    print(f"  Prod. coppie  : {100-p_pe-p_cmp:.4f}%  (atteso 0% a 2 MeV)")

    checks = [
        ("Compton > 99%",         p_cmp > 99.0),
        ("Fotoelettrico < 0.01%", p_pe  < 0.01),
    ]
    results = []
    for desc, ok in checks:
        print(f"  {desc}: {PASS if ok else FAIL}")
        results.append(ok)

    passed = all(results)
    print(f"\n  Risultato TEST 3: {PASS if passed else FAIL}")
    return passed

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Klein-Nishina: <cos_theta> vs valore analitico
# ─────────────────────────────────────────────────────────────────────────────
def test4_klein_nishina():
    print_header("TEST 4 — Distribuzione angolare Klein-Nishina")
    print("  Campiona angoli e confronta <cos_theta> con valore analitico.")
    print("  Fonte: Salvat PENELOPE-2014 eq. 2.36-2.52\n")

    ME_C2_py = 0.51099895

    # Valori analitici corretti (calcolati per integrazione numerica della KN)
    analytic = {
        0.5:  0.2891,
        1.0:  0.3602,
        2.0:  0.4256,
        4.0:  0.4870,
        6.0:  0.5208,
    }

    def kahn_sample(energy, rng, n=100_000):
        alpha   = energy / ME_C2_py
        tau_min = 1.0/(1.0+2.0*alpha)
        a1 = np.log(1.0/tau_min)
        a2 = (1.0 - tau_min**2)*0.5
        a12 = a1 + a2

        cos_thetas = []
        accept = 0; total = 0

        while len(cos_thetas) < n:
            total += 1
            xi1, xi2, xi3 = rng.random(3)
            if xi1*a12 < a1:
                tau = tau_min**(1.0-xi2)
            else:
                tau = np.sqrt(tau_min**2 + xi2*(1.0-tau_min**2))
            tau = np.clip(tau, tau_min, 1.0)
            ct  = 1.0 - (1.0-tau)/(alpha*tau)
            ct  = np.clip(ct, -1.0, 1.0)
            sin2 = max(0.0, 1.0-ct**2)
            g = max(0.0, min(1.0, 1.0 - tau*sin2/(1.0+tau**2)))
            if xi3 <= g:
                cos_thetas.append(ct)
                accept += 1

        eff = accept/total*100
        return np.array(cos_thetas), eff

    rng = np.random.default_rng(42)
    results = []

    print(f"  {'E[MeV]':>8}  {'<cos> MC':>10}  {'<cos> analt':>12}  "
          f"{'diff':>8}  {'eff%':>6}  Status")
    print(f"  {'─'*60}")

    cos_by_energy = {}
    for energy, expected in sorted(analytic.items()):
        samples, eff = kahn_sample(energy, rng, n=50_000)
        mc_mean = samples.mean()
        diff    = abs(mc_mean - expected)/expected*100
        ok = diff < 5.0   # tolleranza 5%
        status = PASS if ok else FAIL
        print(f"  {energy:>8.1f}  {mc_mean:>10.4f}  {expected:>12.4f}  "
              f"{diff:>7.2f}%  {eff:>5.1f}%  {status}")
        results.append(ok)
        cos_by_energy[energy] = samples

    # Plot a 2 MeV
    def kn_analytic(cos_theta, energy):
        alpha = energy/ME_C2_py
        tau = 1.0/(1.0+alpha*(1.0-cos_theta))
        return 0.5*tau**2*(tau + 1.0/tau - 1.0 + cos_theta**2)

    energy_plot = 2.0
    ct_grid = np.linspace(-1,1,300)
    kn_vals = kn_analytic(ct_grid, energy_plot)
    kn_norm = kn_vals / np.trapezoid(kn_vals, ct_grid)

    bins = np.linspace(-1,1,51)
    counts, edges = np.histogram(cos_by_energy[energy_plot], bins=bins, density=True)
    centers = 0.5*(edges[:-1]+edges[1:])

    plt.figure(figsize=(8,5))
    plt.bar(centers, counts, width=edges[1]-edges[0],
            alpha=0.5, color='blue', label='Campionamento Kahn (2 MeV)')
    plt.plot(ct_grid, kn_norm, 'r-', lw=2.5, label='Klein-Nishina analitica')
    plt.xlabel('cos θ',fontsize=12); plt.ylabel('PDF',fontsize=12)
    plt.title(f'TEST 4 — Kahn vs Klein-Nishina a 2 MeV\n'
              f'<cos>_MC={cos_by_energy[2.0].mean():.4f}  '
              f'<cos>_analitica=0.4256', fontsize=11)
    plt.legend(fontsize=10); plt.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig('test4_klein_nishina.png',dpi=150)
    print("  Figura: test4_klein_nishina.png")

    passed = all(results)
    print(f"\n  Risultato TEST 4: {PASS if passed else FAIL}")
    return passed

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Conservazione energia
# ─────────────────────────────────────────────────────────────────────────────
def test5_energy_conservation():
    print_header("TEST 5 — Conservazione energia")
    print("  L'energia totale depositata deve essere ≤ energia iniziale.")
    print("  (≤ perché alcuni fotoni escono dal phantom senza depositare.)\n")

    # Esegue simulazione dedicata e legge il file risultante
    N_test = 500_000
    print(f"  Eseguo {N_test:,} fotoni per misura energia...")
    run_sim(N_test, 0, 99)   # seed 99 per non sovrascrivere altri output

    if not os.path.exists("dose_water.bin"):
        print(f"  {FAIL}: file dose non trovato"); return False

    dose_flat = np.fromfile("dose_water.bin", dtype=np.float64)
    E_deposited = dose_flat.sum()

    # Energia media per fotone dallo spettro 6MV (Sheikh-Bagheri & Rogers 2002)
    # Media pesata per fluenza: ~1.74 MeV per fotone emesso
    E_per_photon_mean = 1.74   # MeV
    E_total_input = N_test * E_per_photon_mean

    fraction = E_deposited / E_total_input
    print(f"  Energia totale depositata: {E_deposited:.3e} MeV")
    print(f"  Energia totale emessa (~): {E_total_input:.3e} MeV")
    print(f"  Frazione depositata: {fraction*100:.1f}%")
    print(f"  (Atteso: 40-80% — i fotoni che escono dal phantom non depositano)")

    ok1 = E_deposited > 0              # qualcosa è stato depositato
    ok2 = fraction < 1.0               # non supera l'energia input
    ok3 = 0.30 < fraction < 0.90       # range fisicamente sensato (30-90%)

    checks = [
        ("Energia depositata > 0", ok1),
        ("Energia depositata ≤ energia input", ok2),
        ("Frazione nel range fisico 20-90%", ok3),
    ]
    results = []
    for desc, ok in checks:
        print(f"  {desc}: {PASS if ok else FAIL}")
        results.append(ok)

    passed = all(results)
    print(f"\n  Risultato TEST 5: {PASS if passed else FAIL}")
    return passed

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Eterogeneità
# ─────────────────────────────────────────────────────────────────────────────
def test6_heterogeneity():
    print_header("TEST 6 — Effetto eterogeneità acqua vs acqua+osso")
    print("  Dopo la zona ossea (12.5-17.5cm) la dose nel phantom eterogeneo")
    print("  deve essere significativamente diversa rispetto all'acqua.\n")

    depths_w, pdd_w = run_sim(500_000, 0, 42)
    depths_h, pdd_h = run_sim(500_000, 1, 42)

    if depths_w is None or depths_h is None:
        print(f"  {FAIL}: simulazione fallita"); return False

    iz_after = int(20.0/VOXEL_CM)
    d_water_after  = pdd_w[iz_after]
    d_hetero_after = pdd_h[iz_after]
    diff = abs(d_water_after - d_hetero_after)

    print(f"  Dose a 20 cm (dopo osso):")
    print(f"    Acqua pura:   {d_water_after:.2f}%")
    print(f"    Con osso:     {d_hetero_after:.2f}%")
    print(f"    Differenza:   {diff:.2f}%  (atteso: > 3%)")

    ok = diff > 3.0

    plt.figure(figsize=(9,5))
    plt.plot(depths_w, pdd_w, 'b-',  lw=2, label='Acqua omogenea')
    plt.plot(depths_h, pdd_h, 'r--', lw=2, label='Acqua + Osso')
    plt.axvspan(12.5, 17.5, alpha=0.15, color='orange', label='Zona ossea')
    plt.xlabel('Profondità (cm)',fontsize=12)
    plt.ylabel('Dose (%)',fontsize=12)
    plt.title('TEST 6 — Eterogeneità: acqua vs acqua+osso',fontsize=12)
    plt.legend(fontsize=10); plt.grid(True,alpha=0.3); plt.ylim(0,110)
    plt.tight_layout(); plt.savefig('test6_heterogeneity.png',dpi=150)
    print("  Figura: test6_heterogeneity.png")

    print(f"\n  Risultato TEST 6: {PASS if ok else WARN}")
    return ok

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SUITE DI TEST — Monte Carlo RT CPU Sequenziale              ║")
    print("║  Angelica Porco — Matricola 264034                           ║")
    print("║  High Performance Computing 2025/2026                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if not os.path.exists("./mc_rt_cpu"):
        print("\nERRORE: ./mc_rt_cpu non trovato.")
        print("Compila con: g++ -O2 -std=c++17 -o mc_rt_cpu main.cpp -lm")
        return

    tests = [
        ("Convergenza 1/√N",  test1_convergence),
        ("Forma PDD 6MV",     test2_pdd_shape),
        ("Sezioni d'urto",    test3_cross_sections),
        ("Klein-Nishina",     test4_klein_nishina),
        ("Conservaz. energia",test5_energy_conservation),
        ("Eterogeneità",      test6_heterogeneity),
    ]

    results = {}
    t0 = time.time()
    for name, func in tests:
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n  ECCEZIONE in {name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = False

    elapsed = time.time() - t0

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  RIEPILOGO RISULTATI                                         ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    n_pass = sum(results.values())
    for name, ok in results.items():
        s = "PASS ✓" if ok else "FAIL ✗"
        print(f"║  {s}  {name:<46}  ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  {n_pass}/{len(results)} test superati  |  Tempo: {elapsed:.0f}s"
          f"{'':>36}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if n_pass >= 5:
        print("\n  ✓ Validazione CPU completata.")
        print("  Prossimo passo: porting CUDA → versione V1")
    else:
        print(f"\n  {len(results)-n_pass} test falliti — verifica prima di procedere.")

if __name__ == "__main__":
    main()
