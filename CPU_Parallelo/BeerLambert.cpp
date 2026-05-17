
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

// TRASPORTO FOTONE SEMPLIFICATO PER BEER LAMBERT
void trasporto_fotoni(Fotone p, const int *phantom, double *dose, Xoshiro256 &rng) {
    while (p.energia > ECUT && verifica_confini(p.x, p.y, p.z)) {
        int mat = phantom[phantom_idx((int)(p.x / VOXEL_CM), (int)(p.y / VOXEL_CM), (int)(p.z / VOXEL_CM))];
        double mu = calcolo_attenuazione_totale(p.energia, mat);
        double d = -std::log(rng()) / mu;

        p.x += p.ux * d;
        p.y += p.uy * d;
        p.z += p.uz * d;

        if (verifica_confini(p.x, p.y, p.z)) {
            int ix = (int)(p.x / VOXEL_CM);
            int iy = (int)(p.y / VOXEL_CM);
            int iz = (int)(p.z / VOXEL_CM);

            dose[phantom_idx(ix, iy, iz)] += p.energia;

            break;
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
            bool stampa_si = (fotoni_completati % intervallo_stampa) < cfg.numero_thread;

            if (stampa_si) {
                auto ora = std::chrono::high_resolution_clock::now();
                double secondi_trascorsi = std::chrono::duration<double>(ora - cfg.istante_inizio).count();
                double fotoni_al_secondo = fotoni_completati / secondi_trascorsi;
                double secondi_rimanenti = (cfg.numero_fotoni - fotoni_completati) / fotoni_al_secondo;

                printf(" [%5.1f%%]  %.0f fotoni/s  ETA %.0fs\n",
                       100.0 * fotoni_completati / cfg.numero_fotoni,
                       fotoni_al_secondo,
                       secondi_rimanenti);
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
    if (tipo_phantom == 0) {
        phantom_label = "Acqua omogenea";
    } else {
        phantom_label = "Acqua + Osso";
    }

    printf("  Monte Carlo per Radioterapia — CPU Parallelo (Beer-Lambert)\n");
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
    double *profilo_dose = new double[NX];
    double *coordinate_cm_laterali = new double[NX];

    if (tipo_phantom == 0) {
        printf("Costruzione phantom con acqua \n");
        costruzione_phantom_acqua(phantom);
    } else {
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
        pdd_file = "./CPU_Parallelo/pdd_water_BL.csv";
        profilo_file = "./CPU_Parallelo/profile_water_BL.csv";
        slice_file = "./CPU_Parallelo/dose_slice_water_BL.csv";
        bin_file = "./CPU_Parallelo/dose_water_BL.bin";
    } else {
        pdd_file = "./CPU_Parallelo/pdd_hetero_BL.csv";
        profilo_file = "./CPU_Parallelo/profile_hetero_BL.csv";
        slice_file = "./CPU_Parallelo/dose_slice_hetero_BL.csv";
        bin_file = "./CPU_Parallelo/dose_hetero_BL.bin";
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
    snprintf(log_file, sizeof(log_file), "logs/CPU_PAR_BL_%d.log", tipo_phantom);

    FILE *f = fopen(log_file, "a");
    if (f) {
        fprintf(f, "TIMING version=CPU_PAR_BL_%d n_fotoni=%lld t_sec=%.6f\n",
                tipo_phantom, num_fotoni, tempo_esecuzione);
        fclose(f);
    }

    printf("  Simulazione completata.\n");
    printf("  Tempo totale: %.3f s  |  Throughput: %.0f fotoni/s\n", tempo_esecuzione, num_fotoni / tempo_esecuzione);

    return 0;
}
