
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "physics.cuh"
#include "compton.cuh"
#include "phantom.cuh"
#include "output.cuh"

#define CUDA_CHECK(operazione)                                                  \
    do {                                                                        \
        cudaError_t risultato = (operazione);                                         \
        if (risultato != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(risultato));         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)



struct Fotone {
    double x, y, z;
    double ux, uy, uz;
    double energia;
};

__device__ inline double genera_rng(curandStatePhilox4_32_10_t *stato) {
    double valore;
    do {
        valore = curand_uniform_double(stato);
    } while (valore >= 1.0);
    return valore;
}

__constant__ double FLUENZA[NUMERO_BINS_SPETTRO];

__device__ inline double campiona_energia(curandStatePhilox4_32_10_t *stato_casuale) {
    double punto_campionamento = genera_rng(stato_casuale);

    int indice_limite_inferiore = 0;
    int indice_limite_superiore = NUMERO_BINS_SPETTRO - 1;
    while (indice_limite_inferiore < indice_limite_superiore) {
        int punto_centrale = (indice_limite_inferiore + indice_limite_superiore) / 2;
        if (FLUENZA[punto_centrale] < punto_campionamento)
            indice_limite_inferiore = punto_centrale + 1;
        else
            indice_limite_superiore = punto_centrale;
    }

    double energia_centrale = ENERGIE_SPETTRO[indice_limite_inferiore];
    double offset = (genera_rng(stato_casuale) - 0.5) * 0.25;
    double energia = energia_centrale + offset;
    if (energia < 0.01)
        energia = 0.01;
    if (energia > 6.00)
        energia = 6.00;
    return energia;
}

__device__ inline Fotone genera_fotone_iniziale(curandStatePhilox4_32_10_t *stato_casuale) {
    double cx = PHANTOM_CM / 2.0;
    double cy = PHANTOM_CM / 2.0;

    Fotone p;
    p.x = cx + (curand_uniform_double(stato_casuale) * 2.0 - 1.0) * SEMI_AMPIEZZA_CAMPO;
    p.y = cy + (curand_uniform_double(stato_casuale) * 2.0 - 1.0) * SEMI_AMPIEZZA_CAMPO;
    p.z = 1.0e-7;
    p.ux = 0.0;
    p.uy = 0.0;
    p.uz = 1.0;
    p.energia = campiona_energia(stato_casuale);
    return p;
}

__device__ inline double distanza_limite_voxel( double x, double y, double z, double ux, double uy, double uz, int ix, int iy, int iz) {
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
            distanza_minima_confine = fmin(distanza_minima_confine, distanza_lineare);
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
            distanza_minima_confine = fmin(distanza_minima_confine, distanza_lineare);
        }
    }
    if (std::fabs(uz) > 1.0e-12) {
        double confine_voxel_Z;
        if (uz > 0){
            confine_voxel_Z = (iz + 1) * VOXEL_CM;
        }
        else{
            confine_voxel_Z = iz * VOXEL_CM;
        }
        double distanza_lineare = (confine_voxel_Z - z) / uz;
        if (distanza_lineare > 1.0e-10) {
            distanza_minima_confine = fmin(distanza_minima_confine, distanza_lineare);
        }
    }
    return distanza_minima_confine;
}

__device__ void trasporto_fotoni( Fotone fotone_iniziale, const int *phantom, double *dose, curandStatePhilox4_32_10_t *stato_casuale)
{
    Fotone stack[64];
    int num_particelle_stack = 0;
    stack[num_particelle_stack++] = fotone_iniziale;

    while (num_particelle_stack > 0) {
        Fotone particella_corrente = stack[--num_particelle_stack];
        for (int step = 0; step < 100000; step++) {
            if (particella_corrente.energia < ECUT) {
                if (verifica_confini(particella_corrente.x, particella_corrente.y, particella_corrente.z)) {
                    int id = phantom_idx(vox(particella_corrente.x), vox(particella_corrente.y), vox(particella_corrente.z));
                    atomicAdd(&dose[id], particella_corrente.energia);
                }
                break;
            }

            if (!verifica_confini(particella_corrente.x, particella_corrente.y, particella_corrente.z))
                break;
            int ix = vox(particella_corrente.x);
            int iy = vox(particella_corrente.y);
            int iz = vox(particella_corrente.z);
            int materiale = phantom[phantom_idx(ix, iy, iz)];
            double mu = calcolo_attenuazione_totale(particella_corrente.energia, materiale);
            if (mu <= 0.0)
                break;

            // Campiona cammino libero medio
            double xi = genera_rng(stato_casuale);
            double distanza_teorica = -log(xi) / mu;
            double distanza_fisica = distanza_limite_voxel(particella_corrente.x, particella_corrente.y, particella_corrente.z, particella_corrente.ux, particella_corrente.uy, particella_corrente.uz, ix, iy, iz);

            if (distanza_teorica <= distanza_fisica) {
                particella_corrente.x += particella_corrente.ux * distanza_teorica;
                particella_corrente.y += particella_corrente.uy * distanza_teorica;
                particella_corrente.z += particella_corrente.uz * distanza_teorica;

                if (!verifica_confini(particella_corrente.x, particella_corrente.y, particella_corrente.z))
                    break;

                ix = vox(particella_corrente.x);
                iy = vox(particella_corrente.y);
                iz = vox(particella_corrente.z);
                materiale = phantom[phantom_idx(ix, iy, iz)];
                int id = phantom_idx(ix, iy, iz);

                int tipo_interazione = seleziona_tipo_interazione(particella_corrente.energia, materiale, genera_rng(stato_casuale));

                if (tipo_interazione == 0) {
                    atomicAdd(&dose[id], particella_corrente.energia);
                    break;
                }
                // -------- COMPTON (Kahn) --------
                else if (tipo_interazione == 1) {
                    double cos_theta;
                    double energia_scatter;
                    while (true) {
                        double xi1 = genera_rng(stato_casuale);
                        double xi2 = genera_rng(stato_casuale);
                        double xi3 = genera_rng(stato_casuale);
                        metodo_kahn(particella_corrente.energia, xi1, xi2, xi3, cos_theta, energia_scatter);
                        if (cos_theta <= 1.0) break;
                    }

                    double energia_ceduta = particella_corrente.energia - energia_scatter;

                    if (energia_ceduta > 0.0) {
                        atomicAdd(&dose[id], energia_ceduta);
                    }

                    particella_corrente.energia = energia_scatter;
                    double phi = 2.0 * PI * genera_rng(stato_casuale);
                    aggiornamento_traiettoria(particella_corrente.ux, particella_corrente.uy, particella_corrente.uz, cos_theta, phi);

                    if (particella_corrente.energia < ECUT) {
                        atomicAdd(&dose[id], particella_corrente.energia);
                        break;
                    }
                }
                else {
                    double energia_cinetica_residua = particella_corrente.energia - 2.0 * ME_C2;
                    if (energia_cinetica_residua > 0.0) {
                        atomicAdd(&dose[id], energia_cinetica_residua);
                    }

                    if (ME_C2 > ECUT && num_particelle_stack + 2 <= 62) {
                        double cos_theta = 2.0 * genera_rng(stato_casuale) - 1.0;
                        double phi_a  = 2.0 * PI * genera_rng(stato_casuale);
                        double sen_theta = sqrt(fmax(0.0, 1.0 - cos_theta * cos_theta));

                        Fotone fotone_secondario_1, fotone_secondario_2;
                        fotone_secondario_1.x = fotone_secondario_2.x = particella_corrente.x;
                        fotone_secondario_1.y = fotone_secondario_2.y = particella_corrente.y;
                        fotone_secondario_1.z = fotone_secondario_2.z = particella_corrente.z;
                        fotone_secondario_1.ux =  sen_theta * cos(phi_a);
                        fotone_secondario_1.uy =  sen_theta * sin(phi_a);
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
                const double eps = 1.0e-7;
                particella_corrente.x += particella_corrente.ux * (distanza_fisica + eps);
                particella_corrente.y += particella_corrente.uy * (distanza_fisica + eps);
                particella_corrente.z += particella_corrente.uz * (distanza_fisica + eps);
            }

        }
    }
}

__global__ void mc_kernel(long long num_fotoni, const int *phantom, double *dose, unsigned long long *contatori_fotoni_procesati, uint64_t seed_base){
    while (true) {
        unsigned long long id_fotone = atomicAdd(contatori_fotoni_procesati, 1ULL);
        if ((long long)id_fotone >= num_fotoni)
            break;
        curandStatePhilox4_32_10_t stato_rng;
        curand_init(seed_base, id_fotone, 0, &stato_rng);
        Fotone p = genera_fotone_iniziale(&stato_rng);
        trasporto_fotoni(p, phantom, dose, &stato_rng);
    }
}

static void calcola_fluenza(double distribuzione_cumulata[NUMERO_BINS_SPETTRO]) {
    static const double FLUENZA[NUMERO_BINS_SPETTRO] = {
        0.0243, 0.0676, 0.0862, 0.0929, 0.0919, 0.0868, 0.0794, 0.0712,
        0.0628, 0.0548, 0.0471, 0.0399, 0.0334, 0.0276, 0.0224, 0.0178,
        0.0138, 0.0104, 0.0075, 0.0052, 0.0034, 0.0020, 0.0010, 0.0004
    };
    double somma_totale = 0.0;
    for (int i = 0; i < NUMERO_BINS_SPETTRO; i++)
        somma_totale += FLUENZA[i];
    distribuzione_cumulata[0] = FLUENZA[0] / somma_totale;
    for (int i = 1; i < NUMERO_BINS_SPETTRO; i++)
        distribuzione_cumulata[i] = distribuzione_cumulata[i-1] + FLUENZA[i] / somma_totale;
    distribuzione_cumulata[NUMERO_BINS_SPETTRO-1] = 1.0;
}

int main(int argc, char *argv[]) {

    long long num_fotoni = 1000000;
    int tipo_phantom = 0;
    uint64_t seed = 42ULL;

    if (argc > 1) num_fotoni = std::atoll(argv[1]);
    if (argc > 2) tipo_phantom = std::atoi(argv[2]);
    if (argc > 3) seed = (uint64_t)std::atoll(argv[3]);

    const char *phantom_label;
    if (tipo_phantom == 0){
      phantom_label = "Acqua omogenea";
    } else{
      phantom_label = "Acqua + Osso";
    }

    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    printf("  Monte Carlo per Radioterapia — GPU CUDA  V2\n\n");
    printf("  GPU        : %s  (SM %d.%d)\n", properties.name, properties.major, properties.minor);
    printf("  Phantom    : %dx%dx%d voxel  |  voxel %.0fmm  |  %.0f³ cm³\n", NX, NY, NZ, VOXEL_CM * 10.0, PHANTOM_CM);
    printf("  Materiale  : %s\n", phantom_label);
    printf("  N fotoni   : %lld\n", num_fotoni);
    printf("  Seed       : %llu\n", (unsigned long long)seed);
    printf("  ECUT       : %.0f keV\n\n", ECUT * 1000.0);

    int *host_phantom = new int[NX * NY * NZ];

    if (tipo_phantom == 0) {
        printf("Costruzione phantom con acqua\n");
        costruzione_phantom_acqua(host_phantom);
    } else {
        printf("Costruzione phantom eterogeneo\n");
        costruzione_phantom_acquaosso(host_phantom);
    }

    int *device_phantom;
    double *device_dose;

    unsigned long long *contatori_fotoni_procesati;

    CUDA_CHECK(cudaMalloc(&device_phantom, NX * NY * NZ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_dose, NX * NY * NZ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&contatori_fotoni_procesati, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(device_dose, 0, NX * NY * NZ * sizeof(double)));
    CUDA_CHECK(cudaMemset(contatori_fotoni_procesati, 0, sizeof(unsigned long long)));

    double host_distribuizione_cumulata[NUMERO_BINS_SPETTRO];
    calcola_fluenza(host_distribuizione_cumulata);
    CUDA_CHECK(cudaMemcpyToSymbol(FLUENZA, host_distribuizione_cumulata, NUMERO_BINS_SPETTRO * sizeof(double)));

    double *host_dose = new double[NX * NY * NZ];

    const int DIMENSIONE_BLOCCO = 256;

    const int NUMERO_BLOCCHI = 1024;

    printf(" Avvio simulazione GPU V2\n");

    cudaEvent_t inizio_copia, fine_copia, inizio_kernel, fine_kernel, inizio_totale, fine_totale;
    cudaEventCreate(&inizio_copia); cudaEventCreate(&fine_copia);
    cudaEventCreate(&inizio_kernel); cudaEventCreate(&fine_kernel);
    cudaEventCreate(&inizio_totale); cudaEventCreate(&fine_totale);

    cudaEventRecord(inizio_copia);
    CUDA_CHECK(cudaMemcpy(device_phantom, host_phantom, NX * NY * NZ * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(fine_copia);
    cudaEventSynchronize(fine_copia);

    cudaEventRecord(inizio_kernel);
    mc_kernel<<<NUMERO_BLOCCHI, DIMENSIONE_BLOCCO>>>( num_fotoni, device_phantom, device_dose, contatori_fotoni_procesati, seed);
    cudaEventRecord(fine_kernel);
    cudaEventSynchronize(fine_kernel);
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(inizio_totale);
    CUDA_CHECK(cudaMemcpy(host_dose, device_dose, NX * NY * NZ * sizeof(double),cudaMemcpyDeviceToHost));
    cudaEventRecord(fine_totale);
    cudaEventSynchronize(fine_totale);

    float ms_copia_a_gpu = 0.0f, ms_calcolo_gpu = 0.0f, ms_copia_da_gpu = 0.0f;
    cudaEventElapsedTime(&ms_copia_a_gpu, inizio_copia, fine_copia);
    cudaEventElapsedTime(&ms_calcolo_gpu, inizio_kernel, fine_kernel);
    cudaEventElapsedTime(&ms_copia_da_gpu, inizio_totale, fine_totale);

    double tempo_calcolo_secondi = ms_calcolo_gpu / 1000.0;
    double tempo_totale_secondi = (ms_copia_a_gpu + ms_calcolo_gpu + ms_copia_da_gpu) / 1000.0;

    cudaEventDestroy(inizio_copia); cudaEventDestroy(fine_copia);
    cudaEventDestroy(inizio_kernel); cudaEventDestroy(fine_kernel);
    cudaEventDestroy(inizio_totale); cudaEventDestroy(fine_totale);

    stampa_statistiche_dose(host_dose, num_fotoni, tempo_calcolo_secondi);

    double *pdd = new double[NZ];
    double *coordinate_cm = new double[NZ];
    double *profilo_dose = new double[NX];
    double *coordinate_cm_laterali = new double[NX];

    calcolo_pdd(host_dose, pdd, coordinate_cm);
    calcolo_profilo_laterale(host_dose, profilo_dose, coordinate_cm_laterali, 10.0);
    stampa_tabella_pdd(coordinate_cm, pdd, phantom_label);

    char pdd_file[256], profilo_file[256], slice_file[256], bin_file[256];

        if (tipo_phantom == 0) {
        snprintf(pdd_file, sizeof(pdd_file), "./GPU_V2/pdd_water.csv");
        snprintf(profilo_file, sizeof(profilo_file), "./GPU_V2/profile_water.csv");
        snprintf(slice_file, sizeof(slice_file), "./GPU_V2/dose_slice_water.csv");
        snprintf(bin_file, sizeof(bin_file), "./GPU_V2/dose_water.bin");
    } else {
        snprintf(pdd_file, sizeof(pdd_file), "./GPU_V2/pdd_hetero.csv");
        snprintf(profilo_file, sizeof(profilo_file), "./GPU_V2/profile_hetero.csv");
        snprintf(slice_file, sizeof(slice_file), "./GPU_V2/dose_slice_hetero.csv");
        snprintf(bin_file, sizeof(bin_file), "./GPU_V2/dose_hetero.bin");
    }

    salva_pdd_csv(coordinate_cm, pdd, pdd_file);
    salva_profilo_csv(coordinate_cm_laterali, profilo_dose, profilo_file);
    salva_fetta_dose_csv(host_dose, slice_file);
    salva_volume_completo(host_dose, bin_file);

    cudaFree(device_phantom);
    cudaFree(device_dose);
    cudaFree(contatori_fotoni_procesati);
    delete[] host_phantom;
    delete[] host_dose;
    delete[] pdd;
    delete[] coordinate_cm;
    delete[] profilo_dose;
    delete[] coordinate_cm_laterali;

    char log_file[64];
    snprintf(log_file, sizeof(log_file), "logs/GPU_V2_%d.log", tipo_phantom);
    FILE *f = fopen(log_file, "a");
    if (f) {
        fprintf(f, "TIMING version=GPU_V2_%d n_fotoni=%lld "
                   "t_h2d_ms=%.3f t_kernel_ms=%.3f t_d2h_ms=%.3f "
                   "tempo_totale_secondi=%.6f\n",
                tipo_phantom, num_fotoni,
                ms_copia_a_gpu, ms_calcolo_gpu, ms_copia_da_gpu,
                tempo_totale_secondi);
        fclose(f);
    }

    printf("  Simulazione completata.\n");
    printf("  H2D: %.3f ms  |  Kernel: %.3f ms  |  D2H: %.3f ms  |  Totale: %.3f ms\n",
           ms_copia_a_gpu, ms_calcolo_gpu, ms_copia_da_gpu, tempo_totale_secondi * 1000.0);

    return 0;
}
