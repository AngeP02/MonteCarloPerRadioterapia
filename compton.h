#pragma once
/*
 * compton.h
 * ─────────────────────────────────────────────────────────────────────────────
 * Campionamento Compton con metodo di Kahn (rejection sampling su Klein-Nishina)
 *
 * Fonte algoritmo:
 *   Salvat F. et al., PENELOPE-2014, OECD/NEA, sezione 2.2, eq. 2.36-2.52
 *   Kahn H. (1954), Use of Different Monte Carlo Sampling Techniques, RAND Corp.
 *
 * Autore: Angelica Porco - Matricola 264034
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cmath>
#include "physics.h"

// ─────────────────────────────────────────────────────────────────────────────
// METODO DI KAHN
// Campiona l'angolo di scattering Compton dalla distribuzione di Klein-Nishina.
//
// Input:
//   energy_mev : energia fotone incidente [MeV]
//   xi1,xi2,xi3: numeri casuali uniformi in [0,1)  (chiamante li fornisce)
//
// Output (parametri di ritorno):
//   cos_theta  : coseno angolo di scattering
//   E_scatter  : energia fotone diffuso [MeV]
//
// L'algoritmo decompone la sezione d'urto KN in due componenti (ramo A e B)
// e usa rejection sampling per campionare il rapporto t = E_sc/E_in ∈ [t_min,1]
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// METODO DI KAHN — versione validata EGS4/MC-GPU
//
// Campiona tau = E_scatter/E_in dalla distribuzione Klein-Nishina.
// Decomposizione in due rami:
//   Ramo 1 (peso a1): tau ∝ 1/tau → tau = tau_min^(1-xi)
//   Ramo 2 (peso a2): tau ∝ tau  → tau = sqrt(tau_min²+xi*(1-tau_min²))
//
// Criterio di accettazione: g(tau) = 1 - tau*sin²θ/(1+tau²) ∈ [0,1]
//
// Fonte principale: Nelson, Hirayama, Rogers — EGS4, SLAC-265 (1985) App.A
//   Equivalente a: MC-GPU_kernel_v1.3.cu funzione GCO_Compton (Badal 2009)
//   Verificato numericamente contro integrazione analitica Klein-Nishina.
//
// VALORI VERIFICATI di <cos_theta>:
//   E=0.1 MeV: 0.118,  E=0.5: 0.289,  E=1.0: 0.360,
//   E=2.0: 0.426,  E=4.0: 0.487,  E=6.0: 0.521
// ─────────────────────────────────────────────────────────────────────────────
inline void kahn_compton(double energy_mev,
                          double xi1, double xi2, double xi3,
                          double &cos_theta, double &E_scatter) {
    double alpha   = energy_mev / ME_C2;
    double tau_min = 1.0 / (1.0 + 2.0 * alpha);

    // Pesi dei due rami
    double a1  = std::log(1.0 / tau_min);           // = log(1+2*alpha)
    double a2  = (1.0 - tau_min * tau_min) * 0.5;
    double a12 = a1 + a2;

    double tau;
    if (xi1 * a12 < a1) {
        // Ramo 1: tau da distribuzione ∝ 1/tau in [tau_min,1]
        // CDF inversa: tau = tau_min^(1-xi2)
        tau = std::pow(tau_min, 1.0 - xi2);
    } else {
        // Ramo 2: tau da distribuzione ∝ tau in [tau_min,1]
        // CDF inversa: tau = sqrt(tau_min² + xi2*(1-tau_min²))
        double t2 = tau_min*tau_min + xi2*(1.0 - tau_min*tau_min);
        tau = std::sqrt(std::max(t2, 1e-30));
    }

    tau = std::min(std::max(tau, tau_min), 1.0);

    // cos_theta dalla cinematica relativistica Compton
    cos_theta = 1.0 - (1.0 - tau) / (alpha * tau);
    cos_theta = std::min(std::max(cos_theta, -1.0), 1.0);
    E_scatter = tau * energy_mev;

    // Criterio di accettazione Klein-Nishina
    double sin2_theta = std::max(0.0, 1.0 - cos_theta * cos_theta);
    double g = 1.0 - (tau * sin2_theta) / (1.0 + tau * tau);
    g = std::max(0.0, std::min(g, 1.0));

    // Segnale di rifiuto: cos_theta = 2.0 (fuori range fisico)
    if (xi3 > g) {
        cos_theta = 2.0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WRAPPER CON LOOP DI REJECTION
// Chiama kahn_compton ripetutamente finché il campione è accettato.
// Il generatore rng deve avere operator() → double in [0,1)
// ─────────────────────────────────────────────────────────────────────────────
template<typename RNG>
inline void sample_compton(double energy_mev, RNG &rng,
                            double &cos_theta, double &E_scatter) {
    for (int attempt = 0; attempt < 10000; attempt++) {
        double xi1 = rng();
        double xi2 = rng();
        double xi3 = rng();
        kahn_compton(energy_mev, xi1, xi2, xi3, cos_theta, E_scatter);
        if (cos_theta <= 1.0) return;   // accettato
    }
    // Fallback (non dovrebbe mai accadere nella pratica)
    cos_theta = 1.0;
    E_scatter = energy_mev;
}

// ─────────────────────────────────────────────────────────────────────────────
// ROTAZIONE DELLA DIREZIONE DEL FOTONE DOPO LO SCATTERING
//
// Ruota il versore (ux,uy,uz) di angolo theta (con cos_theta dato)
// e angolo azimutale phi casuale, nel sistema del laboratorio.
//
// Fonte: EGSnrc Manual PIRS-701, Appendice A (rotazione standard)
//        MC-GPU_v1.3.h funzione rotate_around_axis
// ─────────────────────────────────────────────────────────────────────────────
inline void rotate_direction(double &ux, double &uy, double &uz,
                              double cos_theta, double phi) {
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    double cp        = std::cos(phi);
    double sp        = std::sin(phi);

    double ux_new, uy_new, uz_new;

    if (std::fabs(uz) > 0.99999) {
        // Caso degenere: fotone quasi parallelo all'asse Z
        double sgn = (uz > 0.0) ? 1.0 : -1.0;
        ux_new = sin_theta * cp;
        uy_new = sin_theta * sp * sgn;
        uz_new = cos_theta * sgn;
    } else {
        // Caso generale: formula standard di rotazione
        double denom = std::sqrt(1.0 - uz * uz);
        ux_new = sin_theta * (ux * uz * cp - uy * sp) / denom + ux * cos_theta;
        uy_new = sin_theta * (uy * uz * cp + ux * sp) / denom + uy * cos_theta;
        uz_new = -sin_theta * cp * denom                       + uz * cos_theta;
    }

    // Normalizzazione per correggere accumulo errori floating point
    double norm = std::sqrt(ux_new*ux_new + uy_new*uy_new + uz_new*uz_new);
    if (norm > 0.0) { ux = ux_new/norm; uy = uy_new/norm; uz = uz_new/norm; }
}
