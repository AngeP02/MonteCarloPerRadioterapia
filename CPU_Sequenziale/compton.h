#pragma once

#include <cmath>
#include "physics.h"

// Kahn
inline void kahn_compton(double energia_mev, double xi_1, double xi_2, double xi_3, double &cos_theta, double &energia_scatter) {
    double alpha = energia_mev / ME_C2;
    double tau_min = 1.0 / (1.0 + 2.0 * alpha);

    double area_ramo_1  = std::log(1.0 / tau_min);
    double area_ramo_2  = (1.0 - tau_min * tau_min) * 0.5;
    double area_totale = area_ramo_1 + area_ramo_2;
    double tau;

    if (xi_1 * area_totale < area_ramo_1) {
        // si campiona ramo logaritmico
        tau = std::pow(tau_min, 1.0 - xi_2);
    } else {
        // si fa interpolazione lineare e si campiona ramo lineare
        double tau_min_quadro = tau_min * tau_min;
        double tau_quadro = tau_min_quadro + xi_2*(1.0 - tau_min_quadro);
        tau = std::sqrt(std::max(tau_quadro, 1e-30));
    }

    tau = std::min(std::max(tau, tau_min), 1.0);

    cos_theta = 1.0 - (1.0 - tau) / (alpha * tau);
    cos_theta = std::min(std::max(cos_theta, -1.0), 1.0);
    energia_scatter = tau * energia_mev;

    // calcolo della probabilità di accettazione
    double sin2_theta = std::max(0.0, 1.0 - cos_theta * cos_theta);
    double termine_correttivo = (tau * sin2_theta) / (1.0 + tau * tau);
    double probabilita_accettazione = 1.0 - termine_correttivo;
    probabilita_accettazione = std::max(0.0, std::min(probabilita_accettazione, 1.0));

    if (xi_3 > probabilita_accettazione) {
        cos_theta = 2.0;
    }
}

// Wrapper con loop rejection
template<typename RNG>
inline void sample_compton(double energia_mev, RNG &rng, double &cos_theta, double &energia_scatter) {
    while(true){ // loop rejection perchè prima o poi un valore valido viene estratto
        double xi_1 = rng();
        double xi_2 = rng();
        double xi_3 = rng();
        kahn_compton(energia_mev, xi_1, xi_2, xi_3, cos_theta, energia_scatter);
        if (cos_theta <= 1.0)
            return;
    }
}

// Rotazione della direzione del fotone dopo lo scattering
inline void rotate_direction(double &ux, double &uy, double &uz, double cos_theta, double phi) {
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    double cos_phi = std::cos(phi);
    double sen_phi = std::sin(phi);

    double ux_new, uy_new, uz_new;

    if (std::fabs(uz) > 0.99999) { // se viaggia quasi parallelo all'asse Z
        double segno = 1.0;
        if(uz > 0.0){
          segno = 1.0;
        }else{
          segno = -1.0;
        }
        ux_new = sin_theta * cos_phi;
        uy_new = sin_theta * sen_phi * segno;
        uz_new = cos_theta * segno;
    } else {
        double proiezione_xy = std::sqrt(1.0 - uz * uz);
        ux_new = sin_theta * (ux * uz * cos_phi - uy * sen_phi) / proiezione_xy + ux * cos_theta;
        uy_new = sin_theta * (uy * uz * cos_phi + ux * sen_phi) / proiezione_xy + uy * cos_theta;
        uz_new = -sin_theta * cos_phi * proiezione_xy + uz * cos_theta;
    }

    // Normalizzazione
    double norm = std::sqrt(ux_new * ux_new + uy_new * uy_new + uz_new * uz_new);
    if (norm > 0.0){
       ux = ux_new/norm;
       uy = uy_new/norm;
       uz = uz_new/norm;
    }
}
