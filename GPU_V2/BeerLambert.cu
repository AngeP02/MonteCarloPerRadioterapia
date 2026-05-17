
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "physics.cuh"
#include "phantom.cuh"
#include "output.cuh"

#define CUDA_CHECK(operazione)                                                  \
    do {                                                                        \
        cudaError_t risultato = (operazione);                                   \
        if (risultato != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(risultato));         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

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

static void calcola_fluenza(double distribuzione_cumulata[NUMERO_BINS_SPETTRO]) {
    static const double FLUENZA[NUMERO_BINS_SPETTRO] = {
        0.0243, 0.0676, 0.0862, 0.0929, 0.0919, 0.0868, 0.0794, 0.0712,
        0.0628, 0.0548, 0.0471, 0.0399, 0.0334, 0.0276, 0.0224, 0.0178,
        0.0138, 0.0104, 0.0075, 0.0052, 0.0034, 0.0020, 0.0010, 0.0004
    };
    double somma_totale = 0.0;
    for (int i = 0; i < NUMERO_BINS_SPETTRO; i++) somma_totale += FLUENZA[i];
    distribuzione_cumulata[0] = FLUENZA[0] / somma_totale;
    for (int i = 1; i < NUMERO_BINS_SPETTRO; i++)
        distribuzione_cumulata[i] = distribuzione_cumulata[i-1] + FLUENZA[i] / somma_totale;
    distribuzione_cumulata[NUMERO_BINS_SPETTRO-1] = 1.0;
}

__global__ void mc_beer_lambert_kernel( long long num_fotoni, const int *phantom, double *dose, unsigned long long *contatore_fotoni_processati, uint64_t seed_base)
{
    while (true) {
        unsigned long long indice_thread = atomicAdd(contatore_fotoni_processati, 1ULL);
        if ((long long)indice_thread >= num_fotoni)
            break;
        curandStatePhilox4_32_10_t stato_casuale;
        curand_init(seed_base, indice_thread, 0, &stato_casuale);

        double cx = PHANTOM_CM / 2.0;
        double cy = PHANTOM_CM / 2.0;

        double x = cx + (curand_uniform_double(&stato_casuale) * 2.0 - 1.0) * SEMI_AMPIEZZA_CAMPO;
        double y = cy + (curand_uniform_double(&stato_casuale) * 2.0 - 1.0) * SEMI_AMPIEZZA_CAMPO;
        double z = 1.0e-7;
        double ux = 0.0, uy = 0.0, uz = 1.0;
        double energia = campiona_energia(&stato_casuale);

        while (energia > ECUT && verifica_confini(x, y, z)) {
            int ix = vox(x);
            int iy = vox(y);
            int iz = vox(z);
            int materiale = phantom[phantom_idx(ix, iy, iz)];

            double mu = calcolo_attenuazione_totale(energia, materiale);
            double distanza_percorsa = -log(genera_rng(&stato_casuale)) / mu;

            x += ux * distanza_percorsa;
            y += uy * distanza_percorsa;
            z += uz * distanza_percorsa;

            if (verifica_confini(x, y, z)) {
                int id = phantom_idx(vox(x), vox(y), vox(z));
                atomicAdd(&dose[id], energia);
                break;
            }
        }

    }
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

    printf("  Monte Carlo per Radioterapia — GPU CUDA  [Beer-Lambert  V2]\n\n");
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

    int *device_panthom;
    CUDA_CHECK(cudaMalloc(&device_panthom, NX * NY * NZ * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_panthom, host_phantom, NX * NY * NZ * sizeof(int), cudaMemcpyHostToDevice));

    double *device_dose;
    CUDA_CHECK(cudaMalloc(&device_dose, NX * NY * NZ * sizeof(double)));
    CUDA_CHECK(cudaMemset(device_dose, 0, NX * NY * NZ * sizeof(double)));

    double host_distribuizione_cumulata[NUMERO_BINS_SPETTRO];
    calcola_fluenza(host_distribuizione_cumulata);
    CUDA_CHECK(cudaMemcpyToSymbol(FLUENZA, host_distribuizione_cumulata, NUMERO_BINS_SPETTRO * sizeof(double)));

    unsigned long long *contatore_fotoni_processati;
    CUDA_CHECK(cudaMalloc(&contatore_fotoni_processati, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(contatore_fotoni_processati, 0, sizeof(unsigned long long)));


    const int DIMENSIONE_BLOCCO = 256;
    const int NUMERO_BLOCCHI = 1024;

    printf(" Avvio simulazione GPU \n");

    cudaEvent_t inizio_copia, fine_copia, inizio_kernel, fine_kernel, inizio_totale, fine_totale;
    cudaEventCreate(&inizio_copia); cudaEventCreate(&fine_copia);
    cudaEventCreate(&inizio_kernel); cudaEventCreate(&fine_kernel);
    cudaEventCreate(&inizio_totale); cudaEventCreate(&fine_totale);

    cudaEventRecord(inizio_copia);
    CUDA_CHECK(cudaMemcpy(device_panthom, host_phantom, NX * NY * NZ * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(fine_copia);
    cudaEventSynchronize(fine_copia);

    cudaEventRecord(inizio_kernel);
    mc_beer_lambert_kernel<<<NUMERO_BLOCCHI, DIMENSIONE_BLOCCO>>>( num_fotoni, device_panthom, device_dose, contatore_fotoni_processati, seed);
    cudaEventRecord(fine_kernel);
    cudaEventSynchronize(fine_kernel);
    CUDA_CHECK(cudaGetLastError());

    double *host_dose = new double[NX * NY * NZ];

    cudaEventRecord(inizio_totale);
    CUDA_CHECK(cudaMemcpy(host_dose, device_dose, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost));
    cudaEventRecord(fine_totale);
    cudaEventSynchronize(fine_totale);

    float ms_copia_a_gpu = 0.0f, ms_calcolo_gpu = 0.0f, ms_copia_da_gpu = 0.0f;
    cudaEventElapsedTime(&ms_copia_a_gpu,    inizio_copia, fine_copia);
    cudaEventElapsedTime(&ms_calcolo_gpu, inizio_kernel, fine_kernel);
    cudaEventElapsedTime(&ms_copia_da_gpu,    inizio_totale, fine_totale);

    double tempo_calcolo_secondi = ms_calcolo_gpu / 1000.0;
    double tempo_totale_secondi  = (ms_copia_a_gpu + ms_calcolo_gpu + ms_copia_da_gpu) / 1000.0;

    cudaEventDestroy(inizio_copia);
    cudaEventDestroy(fine_copia);


    CUDA_CHECK(cudaMemcpy(host_dose, device_dose, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost));

    stampa_statistiche_dose(host_dose, num_fotoni, tempo_calcolo_secondi);

    double *pdd = new double[NZ];
    double *coordinate_cm = new double[NZ];
    double *profilo_dose = new double[NX];
    double *coordinate_cm_laterali = new double[NX];

    calcolo_pdd(host_dose, pdd, coordinate_cm);
    calcolo_profilo_laterale(host_dose, profilo_dose, coordinate_cm_laterali, 10.0);
    stampa_tabella_pdd(coordinate_cm, pdd, phantom_label);

    const char *pdd_file, *profilo_file, *slice_file, *bin_file;
    if (tipo_phantom == 0) {
        pdd_file = "./GPU_V2/pdd_water_BL.csv";
        profilo_file = "./GPU_V2/profile_water_BL.csv";
        slice_file = "./GPU_V2/dose_slice_water_BL.csv";
        bin_file = "./GPU_V2/dose_water_BL.bin";
    } else {
        pdd_file = "./GPU_V2/pdd_hetero_BL.csv";
        profilo_file = "./GPU_V2/profile_hetero_BL.csv";
        slice_file = "./GPU_V2/dose_slice_hetero_BL.csv";
        bin_file = "./GPU_V2/dose_hetero_BL.bin";
    }

    salva_pdd_csv(coordinate_cm, pdd, pdd_file);
    salva_profilo_csv(coordinate_cm_laterali, profilo_dose, profilo_file);
    salva_fetta_dose_csv(host_dose, slice_file);
    salva_volume_completo(host_dose, bin_file);

    cudaFree(device_panthom);
    cudaFree(device_dose);
    cudaFree(contatore_fotoni_processati);
    delete[] host_phantom;
    delete[] host_dose;
    delete[] pdd;
    delete[] coordinate_cm;
    delete[] profilo_dose;
    delete[] coordinate_cm_laterali;


    char log_file[64];
    snprintf(log_file, sizeof(log_file), "logs/GPU_V2_BL_%d.log", tipo_phantom);

    FILE *f = fopen(log_file, "a");
    if (f) {
        fprintf(f, "TIMING version=GPU_V2_BL_%d n_fotoni=%lld "
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
