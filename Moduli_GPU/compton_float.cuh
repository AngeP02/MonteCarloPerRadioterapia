
#pragma once

#include <cmath>
#include "physics_float.cuh"

__device__ inline void metodo_kahn(double energia_mev, float xi_1, float xi_2, float xi_3, double &cos_theta, double &energia_scatter) {
    double alpha   = energia_mev / ME_C2;
    double tau_min = 1.0 / (1.0 + 2.0 * alpha);

    float area_ramo_1 = (float)log(1.0 / tau_min);
    float area_ramo_2 = (float)((1.0 - tau_min * tau_min) * 0.5);
    float area_totale = area_ramo_1 + area_ramo_2;

    float tau;
    if (xi_1 * area_totale < area_ramo_1) {
        tau = (float)pow(tau_min, 1.0 - (double)xi_2);
    } else {
        float tau_min_quadro = (float)(tau_min * tau_min);
        float tau_quadro = tau_min_quadro + xi_2 * (1.0f - tau_min_quadro);
        tau = sqrtf(fmaxf(tau_quadro, 1e-30f));
    }

    tau = fminf(fmaxf(tau, (float)tau_min), 1.0f);

    cos_theta = 1.0 - (1.0 - (double)tau) / (alpha * (double)tau);
    cos_theta = fmin(fmax(cos_theta, -1.0), 1.0);
    energia_scatter = (double)tau * energia_mev;

    float sin2_theta = fmaxf(0.0f, 1.0f - (float)(cos_theta * cos_theta));
    float termine_correttivo = (tau * sin2_theta) / (1.0f + tau * tau);
    float probabilita_accettazione = fmaxf(0.0f, fminf(1.0f - termine_correttivo, 1.0f));

    if (xi_3 > probabilita_accettazione)
        cos_theta = 2.0;
}

__device__ inline void aggiornamento_traiettoria(double &ux, double &uy, double &uz, double cos_theta, float phi) {
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - (float)(cos_theta * cos_theta)));
    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    float ux_new, uy_new, uz_new;
    float fux = (float)ux;
    float fuy = (float)uy;
    float fuz = (float)uz;
    float fcos = (float)cos_theta;

    if (fabsf(fuz) > 0.99999f) {
        float segno = 1.0f;
        if (fuz > 0.0f){
           segno = 1.0f;
        }else{
           segno = -1.0f;
        }
        ux_new = sin_theta * cos_phi;
        uy_new = sin_theta * sin_phi * segno;
        uz_new = fcos * segno;
    } else {
        float proiezione_xy = sqrtf(1.0f - fuz * fuz);
        ux_new = sin_theta * (fux * fuz * cos_phi - fuy * sin_phi) / proiezione_xy + fux * fcos;
        uy_new = sin_theta * (fuy * fuz * cos_phi + fux * sin_phi) / proiezione_xy + fuy * fcos;
        uz_new = -sin_theta * cos_phi * proiezione_xy + fuz * fcos;
    }

    float norm = sqrtf(ux_new*ux_new + uy_new*uy_new + uz_new*uz_new);
    if (norm > 0.0f) {
        ux = (double)(ux_new / norm);
        uy = (double)(uy_new / norm);
        uz = (double)(uz_new / norm);
    }
}
