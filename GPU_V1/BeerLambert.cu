
/*
 * Monte Carlo per Radioterapia — GPU CUDA  |  Beer-Lambert
 *
 * Versione semplificata: trasporto a singolo step, nessun Compton/PE/pair.
 * Speculare a BeerLambert.cpp (CPU), con:
 *   - 1 thread per particella
 *   - cuRAND Philox 4x32 per generazione numeri casuali thread-safe
 *   - atomicAdd su double per accumulo dose (richiede SM >= 6.0)
 *
 * Compilazione:
 *   nvcc -O3 -arch=sm_75 -std=c++17 -lcurand -o mc_gpu_bl BeerLambert.cu
 *
 * Utilizzo:
 *   ./mc_gpu_bl [n_fotoni] [tipo_phantom] [seed]
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "physics.cuh"
#include "phantom.cuh"
#include "output.cuh"

// ============================================================
// MACRO DI CONTROLLO ERRORI CUDA
// ============================================================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ============================================================
// HELPER RNG — intervallo aperto (0,1) identico a Xoshiro256
// ============================================================
__device__ inline double rng_open(curandStatePhilox4_32_10_t *st) {
    double r;
    do { r = curand_uniform_double(st); } while (r >= 1.0);
    return r;
}

// ============================================================
// SPETTRO 6MV — CDF in constant memory (identica a random.h)
// ============================================================
static const int SPECTRUM_BINS = 24;

__constant__ double d_SPECTRUM_E[SPECTRUM_BINS] = {
    0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
    2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00,
    4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00
};

__constant__ double d_SPECTRUM_CDF[SPECTRUM_BINS];

__device__ inline double sample_energy_dev(curandStatePhilox4_32_10_t *st) {
    double xi = rng_open(st);
    int lo = 0, hi = SPECTRUM_BINS - 1;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (d_SPECTRUM_CDF[m] < xi) lo = m + 1; else hi = m;
    }
    double e = d_SPECTRUM_E[lo] + (rng_open(st) - 0.5) * 0.25;
    if (e < 0.01) e = 0.01;
    if (e > 6.00) e = 6.00;
    return e;
}

// ============================================================
// KERNEL BEER-LAMBERT — 1 thread = 1 fotone
// Logica identica a transport_photon() in BeerLambert.cpp
// ============================================================
__global__ void beer_lambert_kernel(
    long long  num_fotoni,
    const int *phantom,
    double    *dose,
    uint64_t   seed_base)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_fotoni) return;

    // Inizializza RNG Philox per questo thread (sequenza indipendente)
    curandStatePhilox4_32_10_t st;
    curand_init(seed_base, (unsigned long long)tid, 0, &st);

    // ---------- CAMPIONAMENTO SORGENTE ----------
    // Identico a sample_source() in BeerLambert.cpp
    const double FIELD_HALF = 5.0;
    double cx = PHANTOM_CM / 2.0;
    double cy = PHANTOM_CM / 2.0;

    double x  = cx + (curand_uniform_double(&st) * 2.0 - 1.0) * FIELD_HALF;
    double y  = cy + (curand_uniform_double(&st) * 2.0 - 1.0) * FIELD_HALF;
    double z  = 1.0e-7;
    double ux = 0.0, uy = 0.0, uz = 1.0;   // fascio parallelo lungo Z
    double energia = sample_energy_dev(&st);

    // ---------- TRASPORTO BEER-LAMBERT ----------
    // Identico al while() in transport_photon() di BeerLambert.cpp:
    // campiona un singolo cammino libero medio nel materiale corrente,
    // sposta la particella, deposita l'energia nel voxel di arrivo.
    while (energia > ECUT && inside_dev(x, y, z)) {
        int ix  = vox_dev(x);
        int iy  = vox_dev(y);
        int iz  = vox_dev(z);
        int mat = phantom[phantom_idx_dev(ix, iy, iz)];

        double mu = get_mu_total_dev(energia, mat);
        double d  = -log(rng_open(&st)) / mu;

        x += ux * d;
        y += uy * d;
        z += uz * d;

        if (inside_dev(x, y, z)) {
            int id = phantom_idx_dev(vox_dev(x), vox_dev(y), vox_dev(z));
            atomicAdd(&dose[id], energia);
            break;  // un solo step, identico alla CPU
        }
    }
}

// ============================================================
// HOST — costruisce CDF spettro (identico a Spectrum() in random.h)
// ============================================================
static void build_spectrum_cdf(double cdf[SPECTRUM_BINS]) {
    static const double fluence[SPECTRUM_BINS] = {
        0.0243, 0.0676, 0.0862, 0.0929, 0.0919, 0.0868, 0.0794, 0.0712,
        0.0628, 0.0548, 0.0471, 0.0399, 0.0334, 0.0276, 0.0224, 0.0178,
        0.0138, 0.0104, 0.0075, 0.0052, 0.0034, 0.0020, 0.0010, 0.0004
    };
    double sum = 0.0;
    for (int i = 0; i < SPECTRUM_BINS; i++) sum += fluence[i];
    cdf[0] = fluence[0] / sum;
    for (int i = 1; i < SPECTRUM_BINS; i++)
        cdf[i] = cdf[i-1] + fluence[i] / sum;
    cdf[SPECTRUM_BINS-1] = 1.0;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char *argv[]) {

    long long num_fotoni   = 1000000;
    int       tipo_phantom = 0;
    uint64_t  seed         = 42ULL;

    if (argc > 1) num_fotoni   = std::atoll(argv[1]);
    if (argc > 2) tipo_phantom = std::atoi(argv[2]);
    if (argc > 3) seed          = (uint64_t)std::atoll(argv[3]);

    const char *phantom_label = (tipo_phantom == 0) ? "Acqua omogenea" : "Acqua + Osso";

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("  Monte Carlo per Radioterapia — GPU CUDA  [Beer-Lambert]\n\n");
    printf("  GPU        : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Phantom    : %dx%dx%d voxel  |  voxel %.0fmm  |  %.0f³ cm³\n",
           NX, NY, NZ, VOXEL_CM * 10.0, PHANTOM_CM);
    printf("  Materiale  : %s\n", phantom_label);
    printf("  N fotoni   : %lld\n", num_fotoni);
    printf("  Seed       : %llu\n", (unsigned long long)seed);
    printf("  ECUT       : %.0f keV\n\n", ECUT * 1000.0);

    // -------- PHANTOM CPU → GPU --------
    int *h_phantom = new int[NX * NY * NZ];
    if (tipo_phantom == 0) {
        printf("Costruzione phantom con acqua\n");
        build_phantom_water(h_phantom);
    } else {
        printf("Costruzione phantom eterogeneo\n");
        build_phantom_hetero(h_phantom);
    }

    int *d_phantom;
    CUDA_CHECK(cudaMalloc(&d_phantom, NX * NY * NZ * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_phantom, h_phantom,
                          NX * NY * NZ * sizeof(int), cudaMemcpyHostToDevice));

    // -------- DOSE GPU --------
    double *d_dose;
    CUDA_CHECK(cudaMalloc(&d_dose, NX * NY * NZ * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dose, 0, NX * NY * NZ * sizeof(double)));

    // -------- CDF SPETTRO → constant memory --------
    double h_cdf[SPECTRUM_BINS];
    build_spectrum_cdf(h_cdf);
    CUDA_CHECK(cudaMemcpyToSymbol(d_SPECTRUM_CDF, h_cdf,
                                   SPECTRUM_BINS * sizeof(double)));

    // -------- LANCIO KERNEL --------
    const int THREADS = 256;
    int blocks = (int)((num_fotoni + THREADS - 1) / THREADS);

    printf(" Avvio simulazione GPU (Beer-Lambert)\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    beer_lambert_kernel<<<blocks, THREADS>>>(num_fotoni, d_phantom, d_dose, seed);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double t_sec = std::chrono::duration<double>(t1 - t0).count();

    // -------- COPIA DOSE GPU → CPU --------
    double *h_dose = new double[NX * NY * NZ];
    CUDA_CHECK(cudaMemcpy(h_dose, d_dose,
                          NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost));

    // -------- OUTPUT (identico a BeerLambert.cpp) --------
    print_dose_stats(h_dose, num_fotoni, t_sec);

    double *pdd     = new double[NZ];
    double *coord_z = new double[NZ];
    double *profilo = new double[NX];
    double *coord_x = new double[NX];

    compute_pdd(h_dose, pdd, coord_z);
    compute_lateral_profile(h_dose, profilo, coord_x, 10.0);
    print_pdd_table(coord_z, pdd, phantom_label);

    const char *pdd_file, *prof_file, *slice_file, *bin_file;
    if (tipo_phantom == 0) {
        pdd_file   = "./GPU_V1/pdd_water_BL.csv";
        prof_file  = "./GPU_V1/profile_water_BL.csv";
        slice_file = "./GPU_V1/dose_slice_water_BL.csv";
        bin_file   = "./GPU_V1/dose_water_BL.bin";
    } else {
        pdd_file   = "./GPU_V1/pdd_hetero_BL.csv";
        prof_file  = "./GPU_V1/profile_hetero_BL.csv";
        slice_file = "./GPU_V1/dose_slice_hetero_BL.csv";
        bin_file   = "./GPU_V1/dose_hetero_BL.bin";
    }

    save_pdd_csv(coord_z, pdd, pdd_file);
    save_profile_csv(coord_x, profilo, prof_file);
    save_dose_slice_csv(h_dose, slice_file);
    save_dose_binary(h_dose, bin_file);

    cudaFree(d_phantom);
    cudaFree(d_dose);
    delete[] h_phantom;
    delete[] h_dose;
    delete[] pdd;
    delete[] coord_z;
    delete[] profilo;
    delete[] coord_x;

    printf("  Simulazione completata.\n");
    return 0;
}
