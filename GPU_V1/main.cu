
/*
 * Monte Carlo per Radioterapia — GPU CUDA
 *
 * Parallelismo: 1 thread per particella
 * RNG        : cuRAND Philox 4x32 (thread-safe, alta qualità statistica)
 * Dose       : atomicAdd su double (richiede compute capability >= 6.0)
 *
 * Compilazione:
 *   nvcc -O3 -arch=sm_70 -lcurand main.cu -o mc_gpu
 *   (adattare sm_70 alla propria GPU: sm_60 Pascal, sm_75 Turing, sm_80 Ampere)
 *
 * Utilizzo:
 *   ./mc_gpu [n_fotoni] [tipo_phantom] [seed]
 *   tipo_phantom: 0=acqua omogenea, 1=acqua+osso
 */

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
// STRUTTURA PARTICELLA (identica alla versione CPU)
// ============================================================
struct Particle {
    double x, y, z;
    double ux, uy, uz;
    double energia;
};

// ============================================================
// HELPER RNG — intervallo aperto (0,1) identico a Xoshiro256
// ============================================================
// curand_uniform_double restituisce (0,1] — include 1.0, esclude 0.0.
// Xoshiro256::operator() restituisce (0,1) — esclude entrambi gli estremi.
// La differenza è critica in due punti:
//   1. sample_energy: xi=1.0 → sempre ultimo bin (6 MeV) → spettro distorto
//   2. campionamento MFP: -log(1.0)=0 → distanza zero → interazione fittizia
// Questo helper esclude 1.0 rendendosi statisticamente equivalente a Xoshiro256.
__device__ inline double rng_open(curandStatePhilox4_32_10_t *st) {
    double r;
    do { r = curand_uniform_double(st); } while (r >= 1.0);
    return r;
    // P(r==1.0) ≈ 2^-53 → il loop esegue quasi sempre una sola iterazione
}

// ============================================================
// SPETTRO 6MV — dati in constant memory (identici a random.h)
// ============================================================
static const int SPECTRUM_BINS = 24;

__constant__ double d_SPECTRUM_E[SPECTRUM_BINS] = {
    0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
    2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00,
    4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00
};

__constant__ double d_SPECTRUM_CDF[SPECTRUM_BINS];   // riempita dall'host prima del lancio

// Campionamento energia su device
__device__ inline double sample_energy_dev(curandStatePhilox4_32_10_t *rng_state) {
    double xi = rng_open(rng_state);   // (0,1) — evita xi=1.0 → ultimo bin forzato

    // Ricerca binaria sulla CDF
    int lo = 0, hi = SPECTRUM_BINS - 1;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (d_SPECTRUM_CDF[m] < xi) lo = m + 1; else hi = m;
    }

    double e_centro = d_SPECTRUM_E[lo];
    double offset   = (rng_open(rng_state) - 0.5) * 0.25;
    double e        = e_centro + offset;
    if (e < 0.01) e = 0.01;
    if (e > 6.00) e = 6.00;
    return e;
}

// ============================================================
// CAMPIONAMENTO SORGENTE (device)
// ============================================================
__device__ inline Particle sample_source_dev(curandStatePhilox4_32_10_t *rng_state) {
    const double FIELD_HALF = 5.0;
    double cx = PHANTOM_CM / 2.0;
    double cy = PHANTOM_CM / 2.0;

    Particle p;
    p.x = cx + (curand_uniform_double(rng_state) * 2.0 - 1.0) * FIELD_HALF;
    p.y = cy + (curand_uniform_double(rng_state) * 2.0 - 1.0) * FIELD_HALF;
    p.z = 1.0e-7;
    p.ux = 0.0;
    p.uy = 0.0;
    p.uz = 1.0;
    p.energia = sample_energy_dev(rng_state);
    return p;
}

// ============================================================
// DISTANZA AL PROSSIMO CONFINE DI VOXEL (device, logica identica CPU)
// ============================================================
__device__ inline double boundary_distance_dev(
    double x, double y, double z,
    double ux, double uy, double uz,
    int ix, int iy, int iz)
{
    double dmin = 1.0e30;

    if (fabs(ux) > 1.0e-12) {
        double bx = (ux > 0) ? (ix + 1) * VOXEL_CM : ix * VOXEL_CM;
        double d  = (bx - x) / ux;
        if (d > 1.0e-10) dmin = fmin(dmin, d);
    }
    if (fabs(uy) > 1.0e-12) {
        double by = (uy > 0) ? (iy + 1) * VOXEL_CM : iy * VOXEL_CM;
        double d  = (by - y) / uy;
        if (d > 1.0e-10) dmin = fmin(dmin, d);
    }
    if (fabs(uz) > 1.0e-12) {
        double bz = (uz > 0) ? (iz + 1) * VOXEL_CM : iz * VOXEL_CM;
        double d  = (bz - z) / uz;
        if (d > 1.0e-10) dmin = fmin(dmin, d);
    }
    return dmin;
}

// ============================================================
// TRASPORTO FOTONE — CICLO COMPLETO (device, logica identica a main.cpp CPU)
// ============================================================
__device__ void transport_photon_dev(
    Particle fotone_iniziale,
    const int *phantom,
    double    *dose,
    curandStatePhilox4_32_10_t *rng_state)
{
    // Stack locale per fotoni secondari (pair production)
    Particle stack[64];
    int top = 0;
    stack[top++] = fotone_iniziale;

    while (top > 0) {
        Particle p = stack[--top];

        for (int step = 0; step < 100000; step++) {

            // Cutoff energetico
            if (p.energia < ECUT) {
                if (inside_dev(p.x, p.y, p.z)) {
                    int id = phantom_idx_dev(vox_dev(p.x), vox_dev(p.y), vox_dev(p.z));
                    atomicAdd(&dose[id], p.energia);
                }
                break;
            }

            if (!inside_dev(p.x, p.y, p.z)) break;

            int ix  = vox_dev(p.x);
            int iy  = vox_dev(p.y);
            int iz  = vox_dev(p.z);
            int mat = phantom[phantom_idx_dev(ix, iy, iz)];
            double mu = get_mu_total_dev(p.energia, mat);
            if (mu <= 0.0) break;

            // Campiona cammino libero medio — usa rng_open per escludere 0 e 1:
            // -log(0) = +inf, -log(1) = 0 → entrambi producono distanze anomale
            double xi_mfp        = rng_open(rng_state);
            double dist_teorica  = -log(xi_mfp) / mu;
            double dist_fisica   = boundary_distance_dev(p.x, p.y, p.z,
                                                          p.ux, p.uy, p.uz,
                                                          ix, iy, iz);

            if (dist_teorica <= dist_fisica) {
                // Sposta al punto di interazione
                p.x += p.ux * dist_teorica;
                p.y += p.uy * dist_teorica;
                p.z += p.uz * dist_teorica;

                if (!inside_dev(p.x, p.y, p.z)) break;

                ix  = vox_dev(p.x);
                iy  = vox_dev(p.y);
                iz  = vox_dev(p.z);
                mat = phantom[phantom_idx_dev(ix, iy, iz)];
                int id  = phantom_idx_dev(ix, iy, iz);

                int tipo = select_interaction_dev(p.energia, mat,
                               rng_open(rng_state));   // rng_open esclude 1.0: xi=1.0 → pair production fittizia

                // -------- FOTOELETTRICO --------
                if (tipo == 0) {
                    atomicAdd(&dose[id], p.energia);
                    break;
                }
                // -------- COMPTON (Kahn) --------
                else if (tipo == 1) {
                    double cos_theta, e_scatter;
                    // Loop rejection (identico a CPU)
                    while (true) {
                        double xi1 = rng_open(rng_state);
                        double xi2 = rng_open(rng_state);
                        double xi3 = rng_open(rng_state);
                        kahn_compton_dev(p.energia, xi1, xi2, xi3, cos_theta, e_scatter);
                        if (cos_theta <= 1.0) break;
                    }

                    double e_ceduta = p.energia - e_scatter;
                    if (e_ceduta > 0.0) atomicAdd(&dose[id], e_ceduta);

                    p.energia = e_scatter;
                    double phi = 2.0 * PI * rng_open(rng_state);
                    rotate_direction_dev(p.ux, p.uy, p.uz, cos_theta, phi);

                    if (p.energia < ECUT) {
                        atomicAdd(&dose[id], p.energia);
                        break;
                    }
                }
                // -------- PRODUZIONE DI COPPIE --------
                else {
                    double e_cin = p.energia - 2.0 * ME_C2;
                    if (e_cin > 0.0) atomicAdd(&dose[id], e_cin);

                    if (ME_C2 > ECUT && top + 2 <= 62) {
                        double cos_t  = 2.0 * rng_open(rng_state) - 1.0;
                        double phi_a  = 2.0 * PI * rng_open(rng_state);
                        double sin_t  = sqrt(fmax(0.0, 1.0 - cos_t * cos_t));

                        Particle f1, f2;
                        f1.x = f2.x = p.x;
                        f1.y = f2.y = p.y;
                        f1.z = f2.z = p.z;
                        f1.ux =  sin_t * cos(phi_a);
                        f1.uy =  sin_t * sin(phi_a);
                        f1.uz =  cos_t;
                        f2.ux = -f1.ux;
                        f2.uy = -f1.uy;
                        f2.uz = -f1.uz;
                        f1.energia = f2.energia = ME_C2;

                        stack[top++] = f1;
                        stack[top++] = f2;
                    }
                    break;
                }

            } else {
                // Sposta al confine del voxel con piccolo epsilon
                const double eps = 1.0e-7;
                p.x += p.ux * (dist_fisica + eps);
                p.y += p.uy * (dist_fisica + eps);
                p.z += p.uz * (dist_fisica + eps);
            }

        } // end step loop
    } // end stack loop
}

// ============================================================
// KERNEL PRINCIPALE — 1 thread = 1 particella
// ============================================================
__global__ void mc_kernel(
    long long     num_fotoni,
    const int    *phantom,
    double       *dose,
    uint64_t      seed_base)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_fotoni) return;

    // Inizializza stato Philox per questo thread
    // sequence = tid garantisce sequenze indipendenti tra thread
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed_base, (unsigned long long)tid, 0, &rng_state);

    Particle p = sample_source_dev(&rng_state);
    transport_photon_dev(p, phantom, dose, &rng_state);
}

// ============================================================
// KERNEL BEER-LAMBERT SEMPLIFICATO (corrisponde a BeerLambert.cpp)
// ============================================================
__global__ void mc_beer_lambert_kernel(
    long long     num_fotoni,
    const int    *phantom,
    double       *dose,
    uint64_t      seed_base)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_fotoni) return;

    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed_base, (unsigned long long)tid, 0, &rng_state);

    // Campiona particella sorgente
    const double FIELD_HALF = 5.0;
    double cx = PHANTOM_CM / 2.0, cy = PHANTOM_CM / 2.0;

    Particle p;
    p.x = cx + (curand_uniform_double(&rng_state) * 2.0 - 1.0) * FIELD_HALF;
    p.y = cy + (curand_uniform_double(&rng_state) * 2.0 - 1.0) * FIELD_HALF;
    p.z = 1.0e-7;
    p.ux = 0.0; p.uy = 0.0; p.uz = 1.0;
    p.energia = sample_energy_dev(&rng_state);

    // Trasporto Beer-Lambert: un solo step di attenuazione
    while (p.energia > ECUT && inside_dev(p.x, p.y, p.z)) {
        int mat = phantom[phantom_idx_dev(vox_dev(p.x), vox_dev(p.y), vox_dev(p.z))];
        double mu = get_mu_total_dev(p.energia, mat);
        double d  = -log(rng_open(&rng_state)) / mu;   // rng_open: evita log(0)=+inf

        p.x += p.ux * d;
        p.y += p.uy * d;
        p.z += p.uz * d;

        if (inside_dev(p.x, p.y, p.z)) {
            int id = phantom_idx_dev(vox_dev(p.x), vox_dev(p.y), vox_dev(p.z));
            atomicAdd(&dose[id], p.energia);
            break;
        }
    }
}

// ============================================================
// HOST — calcolo CDF spettro (identico a Spectrum() in random.h)
// ============================================================
static void build_spectrum_cdf(double cdf_out[SPECTRUM_BINS]) {
    static const double fluence[SPECTRUM_BINS] = {
        0.0243, 0.0676, 0.0862, 0.0929, 0.0919, 0.0868, 0.0794, 0.0712,
        0.0628, 0.0548, 0.0471, 0.0399, 0.0334, 0.0276, 0.0224, 0.0178,
        0.0138, 0.0104, 0.0075, 0.0052, 0.0034, 0.0020, 0.0010, 0.0004
    };
    double sum = 0.0;
    for (int i = 0; i < SPECTRUM_BINS; i++) sum += fluence[i];
    cdf_out[0] = fluence[0] / sum;
    for (int i = 1; i < SPECTRUM_BINS; i++)
        cdf_out[i] = cdf_out[i-1] + fluence[i] / sum;
    cdf_out[SPECTRUM_BINS-1] = 1.0;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char *argv[]) {

    long long num_fotoni  = 1000000;
    int       tipo_phantom = 0;
    uint64_t  seed         = 42ULL;
    int       use_bl       = 0;   // 0=ciclo completo, 1=Beer-Lambert

    if (argc > 1) num_fotoni   = std::atoll(argv[1]);
    if (argc > 2) tipo_phantom = std::atoi(argv[2]);
    if (argc > 3) seed          = (uint64_t)std::atoll(argv[3]);
    if (argc > 4) use_bl        = std::atoi(argv[4]);  // 4° argomento opzionale

    const char *phantom_label = (tipo_phantom == 0) ? "Acqua omogenea" : "Acqua + Osso";
    const char *mode_label    = use_bl ? "Beer-Lambert semplificato" : "Ciclo completo (Compton+PE+Pair)";

    // Info GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Monte Carlo per Radioterapia — GPU CUDA\n\n");
    printf("  GPU        : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Modalità   : %s\n", mode_label);
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
    CUDA_CHECK(cudaMemcpy(d_phantom, h_phantom, NX * NY * NZ * sizeof(int), cudaMemcpyHostToDevice));

    // -------- DOSE GPU (inizializzata a zero) --------
    double *d_dose;
    CUDA_CHECK(cudaMalloc(&d_dose, NX * NY * NZ * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dose, 0, NX * NY * NZ * sizeof(double)));

    // -------- CDF SPETTRO → constant memory --------
    double h_cdf[SPECTRUM_BINS];
    build_spectrum_cdf(h_cdf);
    CUDA_CHECK(cudaMemcpyToSymbol(d_SPECTRUM_CDF, h_cdf, SPECTRUM_BINS * sizeof(double)));

    // -------- CONFIGURAZIONE KERNEL --------
    const int THREADS_PER_BLOCK = 256;
    long long num_blocks = (num_fotoni + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf(" Avvio simulazione GPU\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    if (use_bl) {
        mc_beer_lambert_kernel<<<(int)num_blocks, THREADS_PER_BLOCK>>>(
            num_fotoni, d_phantom, d_dose, seed);
    } else {
        mc_kernel<<<(int)num_blocks, THREADS_PER_BLOCK>>>(
            num_fotoni, d_phantom, d_dose, seed);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double t_sec = std::chrono::duration<double>(t1 - t0).count();

    // -------- COPIA DOSE GPU → CPU --------
    double *h_dose = new double[NX * NY * NZ];
    CUDA_CHECK(cudaMemcpy(h_dose, d_dose, NX * NY * NZ * sizeof(double), cudaMemcpyDeviceToHost));

    // -------- OUTPUT (identico alla versione CPU) --------
    print_dose_stats(h_dose, num_fotoni, t_sec);

    double *pdd            = new double[NZ];
    double *coord_z        = new double[NZ];
    double *profilo        = new double[NX];
    double *coord_x        = new double[NX];

    compute_pdd(h_dose, pdd, coord_z);
    compute_lateral_profile(h_dose, profilo, coord_x, 10.0);
    print_pdd_table(coord_z, pdd, phantom_label);

    const char *suffix = use_bl ? "_BL" : "";
    char pdd_file[256], prof_file[256], slice_file[256], bin_file[256];

    if (tipo_phantom == 0) {
        snprintf(pdd_file,   sizeof(pdd_file),   "./GPU_V1/pdd_water%s.csv",       suffix);
        snprintf(prof_file,  sizeof(prof_file),  "./GPU_V1/profile_water%s.csv",   suffix);
        snprintf(slice_file, sizeof(slice_file), "./GPU_V1/dose_slice_water%s.csv",suffix);
        snprintf(bin_file,   sizeof(bin_file),   "./GPU_V1/dose_water%s.bin",       suffix);
    } else {
        snprintf(pdd_file,   sizeof(pdd_file),   "./GPU_V1/pdd_hetero%s.csv",       suffix);
        snprintf(prof_file,  sizeof(prof_file),  "./GPU_V1/profile_hetero%s.csv",   suffix);
        snprintf(slice_file, sizeof(slice_file), "./GPU_V1/dose_slice_hetero%s.csv",suffix);
        snprintf(bin_file,   sizeof(bin_file),   "./GPU_V1/dose_hetero%s.bin",       suffix);
    }

    save_pdd_csv(coord_z, pdd, pdd_file);
    save_profile_csv(coord_x, profilo, prof_file);
    save_dose_slice_csv(h_dose, slice_file);
    save_dose_binary(h_dose, bin_file);

    // -------- PULIZIA --------
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
