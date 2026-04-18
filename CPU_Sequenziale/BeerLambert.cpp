
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
    static const double FIELD_HALF = 5.0;   // +-5 cm -> campo 10×10 cm^2
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

// TRASPORTO FOTONE SEMPLIFICATO PER BEER LAMBERT
void transport_photon(Particle p, const int *phantom, double *dose, Xoshiro256 &rng) {
    while (p.energia > ECUT && inside(p.x, p.y, p.z)) {
        int mat = phantom[phantom_idx((int)(p.x/VOXEL_CM), (int)(p.y/VOXEL_CM), (int)(p.z/VOXEL_CM))];
        double mu = get_mu_total(p.energia, mat);
        double d = -std::log(rng()) / mu;

        p.x += p.ux * d;
        p.y += p.uy * d;
        p.z += p.uz * d;

        if (inside(p.x, p.y, p.z)) {
            int ix = (int)(p.x / VOXEL_CM);
            int iy = (int)(p.y / VOXEL_CM);
            int iz = (int)(p.z / VOXEL_CM);

            dose[phantom_idx(ix, iy, iz)] += p.energia;

            break;
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
      pdd_file = "./CPU_Sequenziale/pdd_water_BL.csv";
      profilo_file = "./CPU_Sequenziale/profile_water_BL.csv";
      slice_file = "./CPU_Sequenziale/dose_slice_water_BL.csv";
      bin_file = "./CPU_Sequenziale/dose_water_BL.bin";
    } else{
      pdd_file = "./CPU_Sequenziale/pdd_hetero_BL.csv";
      profilo_file = "./CPU_Sequenziale/profile_hetero_BL.csv";
      slice_file = "./CPU_Sequenziale/dose_slice_hetero_BL.csv";
      bin_file = "./CPU_Sequenziale/dose_hetero_BL.bin";
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
