
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

// stato di una particella sullo stack
struct Particle {
    double x, y, z;    // posizione [cm]
    double ux, uy, uz; // versore direzione (normalizzato)
    double energia;     // energia [MeV]
};

// CAMPIONAMENTO SORGENTE
inline Particle sample_source(const Spectrum &spettro, Xoshiro256 &rng) {
    static const double FIELD_HALF = 5.0;
    double cx = PHANTOM_CM / 2.0;
    double cy = PHANTOM_CM / 2.0;

    Particle p;
    p.x = cx + (rng() * 2.0 - 1.0) * FIELD_HALF;
    p.y = cy + (rng() * 2.0 - 1.0) * FIELD_HALF;
    p.z = 1.0e-7;
    p.ux = 0.0;
    p.uy = 0.0;
    p.uz = 1.0;
    p.energia = spettro.sample(rng);
    return p;
}

// DISTANZA AL PROSSIMO CONFINE DI VOXEL
inline double boundary_distance(double x, double y, double z, double ux, double uy, double uz, int ix, int iy, int iz) {
    double distanza_minima_confine = 1.0e30; // inizializzata a infinito
    if (std::fabs(ux) > 1.0e-12) {
        double confine_voxel_X;
        if (ux > 0){
            confine_voxel_X = (ix + 1) * VOXEL_CM;
        } else{
            confine_voxel_X = ix * VOXEL_CM;
        }
        double distanza_lineare = (confine_voxel_X - x) / ux;
        if (distanza_lineare > 1.0e-10){
            distanza_minima_confine = std::min(distanza_minima_confine, distanza_lineare);
        }
    }
    if (std::fabs(uy) > 1.0e-12) {
        double confine_voxel_Y;
        if (uy > 0){
            confine_voxel_Y = (iy + 1) * VOXEL_CM;
        } else{
            confine_voxel_Y = iy * VOXEL_CM;
        }
        double distanza_lineare = (confine_voxel_Y - y) / uy;
        if (distanza_lineare > 1.0e-10){
            distanza_minima_confine = std::min(distanza_minima_confine, distanza_lineare);
        }
    }
    if (std::fabs(uz) > 1.0e-12) {
        double confine_voxel_Z;
        if (uz > 0){
            confine_voxel_Z = (iz + 1) * VOXEL_CM;
        } else{
            confine_voxel_Z = iz * VOXEL_CM;
        }
        double distanza_lineare = (confine_voxel_Z - z) / uz;
        if (distanza_lineare > 1.0e-10) {
            distanza_minima_confine = std::min(distanza_minima_confine, distanza_lineare);
        }
    }
    return distanza_minima_confine;
}

// TRASPORTO FOTONE CICLO COMPLETO
void transport_photon(Particle fotone_iniziale, const int *phantom, double *dose, Xoshiro256 &rng) {

    Particle stack[64];
    int num_particelle_stack = 0;
    stack[num_particelle_stack++] = fotone_iniziale;

    while (num_particelle_stack > 0) {
        Particle particella_corrente = stack[--num_particelle_stack];
        for (int step = 0; step < 100000; step++) {
            // Cutoff energetico
            if (particella_corrente.energia < ECUT) {
                // Deposita energia residua nel voxel corrente
                if (inside(particella_corrente.x, particella_corrente.y, particella_corrente.z)) {
                    int ix = vox(particella_corrente.x);
                    int iy = vox(particella_corrente.y);
                    int iz = vox(particella_corrente.z);
                    dose[phantom_idx(ix, iy, iz)] += particella_corrente.energia;
                }
                break;
            }
            // Verifica bounds
            if (!inside(particella_corrente.x, particella_corrente.y, particella_corrente.z))
                break;
            int ix = vox(particella_corrente.x);
            int iy = vox(particella_corrente.y);
            int iz = vox(particella_corrente.z);
            int materiale = phantom[phantom_idx(ix, iy, iz)];
            double mu = get_mu_total(particella_corrente.energia, materiale); // coefficiente di attenuazione totale

            if (mu <= 0.0)
                break;
            // Campiona cammino libero medio
            double xi = rng();
            double distanza_teorica = -std::log(xi) / mu;
            double distanza_fisica = boundary_distance(particella_corrente.x, particella_corrente.y, particella_corrente.z, particella_corrente.ux, particella_corrente.uy, particella_corrente.uz, ix, iy, iz);

            if (distanza_teorica <= distanza_fisica) {
                // Sposta la particella al punto di interazione
                particella_corrente.x += particella_corrente.ux * distanza_teorica;
                particella_corrente.y += particella_corrente.uy * distanza_teorica;
                particella_corrente.z += particella_corrente.uz * distanza_teorica;

                if (!inside(particella_corrente.x, particella_corrente.y, particella_corrente.z))
                    break;

                // Ricalcola voxel
                ix = vox(particella_corrente.x);
                iy = vox(particella_corrente.y);
                iz = vox(particella_corrente.z);
                materiale = phantom[phantom_idx(ix, iy, iz)];
                int id_voxel = phantom_idx(ix, iy, iz);
                int tipo_interazione = select_interaction(particella_corrente.energia, materiale, rng());

                // FOTOELETTRICO: assorbimento totale
                if (tipo_interazione == 0) {
                    dose[id_voxel] += particella_corrente.energia;
                    break;
                }
                // COMPTON: metodo di Kahn
                else if (tipo_interazione == 1) {
                    double cos_theta;
                    double energia_scatter;
                    sample_compton(particella_corrente.energia, rng, cos_theta, energia_scatter);

                    // Deposita energia ceduta all'elettrone (KERMA locale)
                    double energia_ceduta = particella_corrente.energia - energia_scatter;
                    if (energia_ceduta > 0.0) {
                        dose[id_voxel] += energia_ceduta;
                    }

                    // Aggiorna energia e direzione del fotone
                    particella_corrente.energia = energia_scatter;
                    double phi = 2.0 * PI * rng();
                    rotate_direction(particella_corrente.ux, particella_corrente.uy, particella_corrente.uz, cos_theta, phi);

                    if (particella_corrente.energia < ECUT) {
                        dose[id_voxel] += particella_corrente.energia;
                        break;
                    }
                }

                // PRODUZIONE DI COPPIE
                else {
                    // Energia cinetica disponibile per elettrone e positrone
                    double energia_cinetica_residua = particella_corrente.energia - 2.0 * ME_C2;
                    if (energia_cinetica_residua > 0.0) {
                        dose[id_voxel] += energia_cinetica_residua;
                    }

                    if (ME_C2 > ECUT && num_particelle_stack + 2 <= 62) {
                        double cos_theta = 2.0 * rng() - 1.0;
                        double phi_a = 2.0 * PI * rng();
                        double sen_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

                        Particle fotone_secondario_1, fotone_secondario_2;
                        fotone_secondario_1.x = fotone_secondario_2.x = particella_corrente.x;
                        fotone_secondario_1.y = fotone_secondario_2.y = particella_corrente.y;
                        fotone_secondario_1.z = fotone_secondario_2.z = particella_corrente.z;
                        fotone_secondario_1.ux =  sen_theta * std::cos(phi_a);
                        fotone_secondario_1.uy =  sen_theta * std::sin(phi_a);
                        fotone_secondario_1.uz =  cos_theta;
                        fotone_secondario_2.ux = -fotone_secondario_1.ux;
                        fotone_secondario_2.uy = -fotone_secondario_1.uy;
                        fotone_secondario_2.uz = -fotone_secondario_1.uz;
                        fotone_secondario_1.energia = fotone_secondario_2.energia = ME_C2;

                        stack[num_particelle_stack++] = fotone_secondario_1;
                        stack[num_particelle_stack++] = fotone_secondario_2;
                    }
                    break;
                }

            } else {
                double eps = 1.0e-7;
                particella_corrente.x += particella_corrente.ux * (distanza_fisica + eps);
                particella_corrente.y += particella_corrente.uy * (distanza_fisica + eps);
                particella_corrente.z += particella_corrente.uz * (distanza_fisica + eps);
            }

        }
    }
}

int main(int argc, char *argv[]) {

    // valori default
    long long num_fotoni = 1000000;   // default: 1M fotoni
    int tipo_phantom = 0;         // 0=acqua, 1=eterogeneo
    uint64_t seed   = 42ULL;

    if (argc > 1) num_fotoni = std::atoll(argv[1]);
    if (argc > 2) tipo_phantom = std::atoi(argv[2]);
    if (argc > 3) seed = (uint64_t)std::atoll(argv[3]);

    const char *phantom_label;
    if (tipo_phantom == 0){
      phantom_label = "Acqua omogenea";
    } else{
      phantom_label = "Acqua + Osso";
    }

    printf("  Monte Carlo per Radioterapia — CPU Sequenziale\n");
    printf("\n  Parametri:\n");
    printf("  Phantom    : %dx%dx%d voxel  |  voxel %.0fmm  |  %.0f³ cm³\n", NX, NY, NZ, VOXEL_CM * 10.0, PHANTOM_CM);
    printf("  Materiale  : %s\n", phantom_label);
    printf("  N fotoni   : %lld\n", num_fotoni);
    printf("  Seed       : %llu\n", (unsigned long long)seed);
    printf("  ECUT       : %.0f keV\n\n", ECUT * 1000.0);

    int *phantom = new int [NX * NY * NZ];
    double *dose = new double[NX * NY * NZ]();
    double *pdd = new double[NZ];
    double *coordinate_cm = new double[NZ];
    double *profilo_dose = new double[NX];
    double *coordinate_cm_laterali = new double[NX];

    if (tipo_phantom == 0){
        printf("Costruzione phantom con acqua \n");
        build_phantom_water(phantom);
    }else {
        printf("Costruzione phantom eterogeneo \n");
        build_phantom_hetero(phantom);
    }

    Spectrum spettro;  // spettro 6MV con CDF
    Xoshiro256 rng(seed); // PRNG xoshiro256**

    printf(" Avvio simulazione \n");
    auto tempo_inizio_esatto = std::chrono::high_resolution_clock::now();

    long long report_step = std::max(1LL, num_fotoni / 20);   // report ogni 5%

    for (long long i = 0; i < num_fotoni; i++) {
        if ((i + 1) % report_step == 0) {
            auto tempo_attuale     = std::chrono::high_resolution_clock::now();
            double tempo_esecuzione = std::chrono::duration<double>(tempo_attuale - tempo_inizio_esatto).count();
            double rate  = (i + 1) / tempo_esecuzione;
            double tempo_necessrio_fotoni_rimanenti   = (num_fotoni - i - 1) / rate;
            printf(" [%5.1f%%]  %.0f fotoni/s  Estimated Time of Arrival %.0fs\n", 100.0 * (i + 1) / num_fotoni, rate, tempo_necessrio_fotoni_rimanenti);
        }

        Particle p = sample_source(spettro, rng);
        transport_photon(p, phantom, dose, rng);
    }

    auto tempo_fine_esatto = std::chrono::high_resolution_clock::now();
    double tempo_esecuzione = std::chrono::duration<double>(tempo_fine_esatto - tempo_inizio_esatto).count();

    print_dose_stats(dose, num_fotoni, tempo_esecuzione);

    compute_pdd(dose, pdd, coordinate_cm);
    compute_lateral_profile(dose, profilo_dose, coordinate_cm_laterali, 10.0);
    print_pdd_table(coordinate_cm, pdd, phantom_label);

    const char *pdd_file;
    const char *profilo_file;
    const char *slice_file;
    const char *bin_file;

    if (tipo_phantom == 0){
      pdd_file = "./CPU_Sequenziale/pdd_water.csv";
      profilo_file = "./CPU_Sequenziale/profile_water.csv";
      slice_file = "./CPU_Sequenziale/dose_slice_water.csv";
      bin_file = "./CPU_Sequenziale/dose_water.bin";
    } else{
      pdd_file = "./CPU_Sequenziale/pdd_hetero.csv";
      profilo_file = "./CPU_Sequenziale/profile_hetero.csv";
      slice_file = "./CPU_Sequenziale/dose_slice_hetero.csv";
      bin_file = "./CPU_Sequenziale/dose_hetero.bin";
    }
    save_pdd_csv(coordinate_cm, pdd, pdd_file);
    save_profile_csv(coordinate_cm_laterali, profilo_dose, profilo_file);
    save_dose_slice_csv(dose, slice_file);
    save_dose_binary(dose, bin_file);

    delete[] phantom;
    delete[] dose;
    delete[] pdd;
    delete[] coordinate_cm;
    delete[] profilo_dose;
    delete[] coordinate_cm_laterali;

    printf("  Simulazione completata.\n");

    return 0;
}
