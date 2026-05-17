
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

#include "physics.h"
#include "compton.h"
#include "random.h"
#include "phantom.h"
#include "output.h"

struct Fotone {
    double x, y, z;
    double ux, uy, uz;
    double energia;
};

inline Fotone genera_fotone_iniziale(const Spettro &spettro, Xoshiro256 &rng) {
    double cx = PHANTOM_CM / 2.0;
    double cy = PHANTOM_CM / 2.0;
    Fotone p;
    p.x = cx + (rng() * 2.0 - 1.0) * SEMI_AMPIEZZA_CAMPO;
    p.y = cy + (rng() * 2.0 - 1.0) * SEMI_AMPIEZZA_CAMPO;
    p.z = 1.0e-7;
    p.ux = 0.0;
    p.uy = 0.0;
    p.uz = 1.0;
    p.energia = spettro.campiona_energia(rng);
    return p;
}

inline double distanza_limite_voxel(double x, double y, double z, double ux, double uy, double uz, int ix, int iy, int iz) {
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

void trasporto_fotoni(Fotone fotone_iniziale, const int *phantom, double *dose, Xoshiro256 &rng) {

    Fotone stack[64];
    int num_particelle_stack = 0;
    stack[num_particelle_stack++] = fotone_iniziale;

    while (num_particelle_stack > 0) {
        Fotone particella_corrente = stack[--num_particelle_stack];
        for (int step = 0; step < 100000; step++) {
            // Cutoff energetico
            if (particella_corrente.energia < ECUT) {
                // Deposita energia residua nel voxel corrente
                if (verifica_confini(particella_corrente.x, particella_corrente.y, particella_corrente.z)) {
                    int ix = vox(particella_corrente.x);
                    int iy = vox(particella_corrente.y);
                    int iz = vox(particella_corrente.z);
                    dose[phantom_idx(ix, iy, iz)] += particella_corrente.energia;
                }
                break;
            }
            // Verifica bounds
            if (!verifica_confini(particella_corrente.x, particella_corrente.y, particella_corrente.z))
                break;
            int ix = vox(particella_corrente.x);
            int iy = vox(particella_corrente.y);
            int iz = vox(particella_corrente.z);
            int materiale = phantom[phantom_idx(ix, iy, iz)];
            double mu = calcolo_attenuazione_totale(particella_corrente.energia, materiale); // coefficiente di attenuazione totale

            if (mu <= 0.0)
                break;
            // Campiona cammino libero medio
            double xi = rng();
            double distanza_teorica = -std::log(xi) / mu;
            double distanza_fisica = distanza_limite_voxel(particella_corrente.x, particella_corrente.y, particella_corrente.z, particella_corrente.ux, particella_corrente.uy, particella_corrente.uz, ix, iy, iz);

            if (distanza_teorica <= distanza_fisica) {
                // Sposta la particella al punto di interazione
                particella_corrente.x += particella_corrente.ux * distanza_teorica;
                particella_corrente.y += particella_corrente.uy * distanza_teorica;
                particella_corrente.z += particella_corrente.uz * distanza_teorica;

                if (!verifica_confini(particella_corrente.x, particella_corrente.y, particella_corrente.z))
                    break;

                // Ricalcola voxel
                ix = vox(particella_corrente.x);
                iy = vox(particella_corrente.y);
                iz = vox(particella_corrente.z);
                materiale = phantom[phantom_idx(ix, iy, iz)];
                int id_voxel = phantom_idx(ix, iy, iz);
                int tipo_interazione = seleziona_tipo_interazione(particella_corrente.energia, materiale, rng());

                // FOTOELETTRICO: assorbimento totale
                if (tipo_interazione == 0) {
                    dose[id_voxel] += particella_corrente.energia;
                    break;
                }
                // COMPTON: metodo di Kahn
                else if (tipo_interazione == 1) {
                    double cos_theta;
                    double energia_scatter;
                    estrazione_compton(particella_corrente.energia, rng, cos_theta, energia_scatter);
                    // Deposita energia ceduta all'elettrone (KERMA locale)
                    double energia_ceduta = particella_corrente.energia - energia_scatter;
                    if (energia_ceduta > 0.0) {
                        dose[id_voxel] += energia_ceduta;
                    }
                    // Aggiorna energia e direzione del fotone
                    particella_corrente.energia = energia_scatter;
                    double phi = 2.0 * PI * rng();
                    aggiornamento_traiettoria(particella_corrente.ux, particella_corrente.uy, particella_corrente.uz, cos_theta, phi);

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

                        Fotone fotone_secondario_1, fotone_secondario_2;
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

struct ConfigurazioneWorker {
    int id_thread;
    int numero_thread;
    long long numero_fotoni;
    uint64_t seed_casuale;
    const int* phantom;
    double* dose_globale;
    std::atomic<long long>& contatore_progresso;
    std::chrono::time_point<std::chrono::high_resolution_clock>& istante_inizio;
};

void worker(const ConfigurazioneWorker& cfg) {
    uint64_t seed_thread = cfg.seed_casuale + (uint64_t)cfg.id_thread * 2654435761ULL;
    Xoshiro256 generatore(seed_thread);

    Spettro spettro;
    std::vector<double> dose_locale(NX * NY * NZ, 0.0);

    for (long long i = cfg.id_thread; i < cfg.numero_fotoni; i += cfg.numero_thread) {
        Fotone fotone = genera_fotone_iniziale(spettro, generatore);
        trasporto_fotoni(fotone, cfg.phantom, dose_locale.data(), generatore);

        if (cfg.id_thread == 0) {
            long long fotoni_completati = cfg.contatore_progresso.fetch_add(
                cfg.numero_thread, std::memory_order_relaxed) + cfg.numero_thread;

            long long intervallo_stampa = std::max(1LL, cfg.numero_fotoni / 20);
            bool stampa_si  = (fotoni_completati % intervallo_stampa) < cfg.numero_thread;

            if (stampa_si) {
                auto ora = std::chrono::high_resolution_clock::now();
                double secondi_trascorsi = std::chrono::duration<double>(ora - cfg.istante_inizio).count();
                double fotoni_al_secondo = fotoni_completati / secondi_trascorsi;
                double secondi_rimanenti = (cfg.numero_fotoni - fotoni_completati) / fotoni_al_secondo;

                printf(" [%5.1f%%]  %.0f fotoni/s  ETA %.0fs\n", 100.0 * fotoni_completati / cfg.numero_fotoni, fotoni_al_secondo, secondi_rimanenti);
            }
        }
    }

    static std::mutex mutex_riduzione;
    {
        std::lock_guard<std::mutex> lock(mutex_riduzione);
        for (int k = 0; k < NX * NY * NZ; k++)
            cfg.dose_globale[k] += dose_locale[k];
    }
}

int main(int argc, char *argv[]) {

    long long num_fotoni = 1000000;
    int tipo_phantom = 0;
    uint64_t seed = 42ULL;
    int num_thread = (int)std::thread::hardware_concurrency();
    if (num_thread < 1)
        num_thread = 1;

    if (argc > 1) num_fotoni = std::atoll(argv[1]);
    if (argc > 2) tipo_phantom = std::atoi(argv[2]);
    if (argc > 3) seed = (uint64_t)std::atoll(argv[3]);
    if (argc > 4) num_thread = std::atoi(argv[4]);

    const char *phantom_label;
    if (tipo_phantom == 0){
        phantom_label = "Acqua omogenea";
    } else{
        phantom_label = "Acqua + Osso";
    }

    printf("  Monte Carlo per Radioterapia — CPU Parallelo\n");
    printf("\n  Parametri:\n");
    printf("  Phantom    : %dx%dx%d voxel  |  voxel %.0fmm  |  %.0f³ cm³\n", NX, NY, NZ, VOXEL_CM * 10.0, PHANTOM_CM);
    printf("  Materiale  : %s\n", phantom_label);
    printf("  N fotoni   : %lld\n", num_fotoni);
    printf("  Seed       : %llu\n", (unsigned long long)seed);
    printf("  ECUT       : %.0f keV\n", ECUT * 1000.0);
    printf("  Thread     : %d\n\n", num_thread);

    int *phantom = new int[NX * NY * NZ];
    double *dose = new double[NX * NY * NZ]();
    double *pdd = new double[NZ];
    double *coordinate_cm = new double[NZ];
    double *profilo_dose  = new double[NX];
    double *coordinate_cm_laterali = new double[NX];

    if (tipo_phantom == 0){
        printf("Costruzione phantom con acqua \n");
        costruzione_phantom_acqua(phantom);
    }else {
        printf("Costruzione phantom eterogeneo \n");
        costruzione_phantom_acquaosso(phantom);
    }

    printf(" Avvio simulazione \n");
    auto tempo_inizio_esatto = std::chrono::high_resolution_clock::now();

    std::atomic<long long> contatore{0};

    std::vector<std::thread> threads;
    threads.reserve(num_thread);
    for (int t = 0; t < num_thread; t++) {
        ConfigurazioneWorker cfg {
            .id_thread = t,
            .numero_thread = num_thread,
            .numero_fotoni = num_fotoni,
            .seed_casuale = seed,
            .phantom = phantom,
            .dose_globale = dose,
            .contatore_progresso = contatore,
            .istante_inizio = tempo_inizio_esatto
        };
        threads.emplace_back(worker, cfg);
    }

    for (auto &th : threads)
        th.join();

    auto tempo_fine_esatto = std::chrono::high_resolution_clock::now();
    double tempo_esecuzione = std::chrono::duration<double>(tempo_fine_esatto - tempo_inizio_esatto).count();


    stampa_statistiche_dose(dose, num_fotoni, tempo_esecuzione);
    calcolo_pdd(dose, pdd, coordinate_cm);
    calcolo_profilo_laterale(dose, profilo_dose, coordinate_cm_laterali, 10.0);
    stampa_tabella_pdd(coordinate_cm, pdd, phantom_label);

    const char *pdd_file;
    const char *profilo_file;
    const char *slice_file;
    const char *bin_file;

    if (tipo_phantom == 0) {
        pdd_file = "./CPU_Parallelo/pdd_water.csv";
        profilo_file = "./CPU_Parallelo/profile_water.csv";
        slice_file = "./CPU_Parallelo/dose_slice_water.csv";
        bin_file = "./CPU_Parallelo/dose_water.bin";
    } else {
        pdd_file = "./CPU_Parallelo/pdd_hetero.csv";
        profilo_file = "./CPU_Parallelo/profile_hetero.csv";
        slice_file = "./CPU_Parallelo/dose_slice_hetero.csv";
        bin_file = "./CPU_Parallelo/dose_hetero.bin";
    }

    salva_pdd_csv(coordinate_cm, pdd, pdd_file);
    salva_profilo_csv(coordinate_cm_laterali, profilo_dose, profilo_file);
    salva_fetta_dose_csv(dose, slice_file);
    salva_volume_completo(dose, bin_file);

    delete[] phantom;
    delete[] dose;
    delete[] pdd;
    delete[] coordinate_cm;
    delete[] profilo_dose;
    delete[] coordinate_cm_laterali;


    char log_file[64];
    snprintf(log_file, sizeof(log_file), "logs/CPU_PAR_%d.log", tipo_phantom);

    FILE *f = fopen(log_file, "a");
    if (f) {
        fprintf(f, "TIMING version=CPU_PAR_%d n_fotoni=%lld t_sec=%.6f\n",
                tipo_phantom, num_fotoni, tempo_esecuzione);
        fclose(f);
    }

    printf("  Simulazione completata.\n");
    printf("  Tempo totale: %.3f s  |  Throughput: %.0f fotoni/s\n", tempo_esecuzione, num_fotoni / tempo_esecuzione);

    return 0;
}
