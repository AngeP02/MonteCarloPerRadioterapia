
#pragma once

#include <cmath>
#include "physics.cuh"

// Kahn rejection sampling (device)
__device__ inline void kahn_compton_dev(
    double energia_mev,
    double xi_1, double xi_2, double xi_3,
    double &cos_theta, double &energia_scatter)
{
    double alpha    = energia_mev / ME_C2;
    double tau_min  = 1.0 / (1.0 + 2.0 * alpha);

    double area_ramo_1 = log(1.0 / tau_min);
    double area_ramo_2 = (1.0 - tau_min * tau_min) * 0.5;
    double area_totale = area_ramo_1 + area_ramo_2;
    double tau;

    if (xi_1 * area_totale < area_ramo_1) {
        tau = pow(tau_min, 1.0 - xi_2);
    } else {
        double tmin2  = tau_min * tau_min;
        double t2     = tmin2 + xi_2 * (1.0 - tmin2);
        tau = sqrt(fmax(t2, 1e-30));
    }

    tau      = fmin(fmax(tau, tau_min), 1.0);
    cos_theta = 1.0 - (1.0 - tau) / (alpha * tau);
    cos_theta = fmin(fmax(cos_theta, -1.0), 1.0);
    energia_scatter = tau * energia_mev;

    double sin2_theta = fmax(0.0, 1.0 - cos_theta * cos_theta);
    double corr       = (tau * sin2_theta) / (1.0 + tau * tau);
    double prob_acc   = fmax(0.0, fmin(1.0 - corr, 1.0));

    if (xi_3 > prob_acc)
        cos_theta = 2.0;  // segnale di rejection
}

// Rotazione della direzione (device)
__device__ inline void rotate_direction_dev(
    double &ux, double &uy, double &uz,
    double cos_theta, double phi)
{
    double sin_theta = sqrt(fmax(0.0, 1.0 - cos_theta * cos_theta));
    double cos_phi   = cos(phi);
    double sin_phi   = sin(phi);

    double ux_new, uy_new, uz_new;

    if (fabs(uz) > 0.99999) {
        double sgn = (uz > 0.0) ? 1.0 : -1.0;
        ux_new = sin_theta * cos_phi;
        uy_new = sin_theta * sin_phi * sgn;
        uz_new = cos_theta * sgn;
    } else {
        double rxy = sqrt(1.0 - uz * uz);
        ux_new = sin_theta * (ux * uz * cos_phi - uy * sin_phi) / rxy + ux * cos_theta;
        uy_new = sin_theta * (uy * uz * cos_phi + ux * sin_phi) / rxy + uy * cos_theta;
        uz_new = -sin_theta * cos_phi * rxy + uz * cos_theta;
    }

    double norm = sqrt(ux_new*ux_new + uy_new*uy_new + uz_new*uz_new);
    if (norm > 0.0) {
        ux = ux_new / norm;
        uy = uy_new / norm;
        uz = uz_new / norm;
    }
}
