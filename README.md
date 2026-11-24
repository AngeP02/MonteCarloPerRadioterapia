# Monte Carlo Radiation Dose Simulation with CUDA

A Monte Carlo photon transport simulator for radiotherapy dose calculation, implemented in C++ and CUDA and developed as part of the High Performance Computing course at the University of Calabria.

The project explores CPU and GPU implementations of a radiation transport simulation, with a focus on parallelization strategies, GPU memory optimization, numerical precision, and physical validation.

## Overview

This project implements a Monte Carlo simulator for radiation dose deposition in homogeneous and heterogeneous voxelized phantoms. The simulated radiation beam is based on a 6 MV Varian 2100C linear accelerator model, using the photon energy spectrum reported by Sheikh-Bagheri and Rogers (2002).

The simulator models photon transport through matter, tracking stochastic interactions and energy deposition within a three-dimensional voxel grid.

The main objective is to develop and compare five implementations, from a sequential CPU baseline to progressively optimized CUDA versions.

### Key results

- **Five implementations:** sequential CPU, parallel CPU, and three CUDA versions.
- **GPU optimization:** selective FP32 arithmetic and 3D texture memory in GPU V3.
- **Cache performance:** L1 cache hit rate increased from 29–34% to 91% in the optimized GPU implementation.
- **Numerical consistency:** fewer than 0.0004% of voxels with non-zero dose showed a difference between GPU V3 and the previous implementations.
- **Physical validation:** a suite of 10 tests covering photon interactions, dose distributions, energy conservation, and statistical convergence.

## Implementations

| Version | Technology | Main approach |
|---|---|---|
| Sequential CPU | C++ | Reference implementation and physical baseline |
| Parallel CPU | C++ / OpenMP | Multithreaded photon transport with round-robin scheduling |
| GPU V1 | CUDA C | One CUDA thread per photon |
| GPU V2 | CUDA C | Persistent thread pool and atomic work queue |
| GPU V3 | CUDA C | Selective FP32 arithmetic and texture memory |

## Physical model

### Photon interactions

The simulator models three fundamental photon interaction processes in the energy range relevant to radiotherapy, using cross-section data from the NIST XCOM database.

- **Photoelectric absorption:** the photon is absorbed and its energy is deposited locally.
- **Compton scattering:** the scattering angle is sampled using the Kahn rejection-sampling method based on the Klein–Nishina distribution. The photon continues with reduced energy and a new direction.
- **Pair production:** the photon generates an electron–positron pair. The positron annihilates, producing two 0.511 MeV photons.

### Photon transport

The distance to the next interaction is sampled by inversion of the exponential attenuation distribution:

$$
s = -\frac{\ln(\xi)}{\mu}
$$
where $\xi$ is a uniformly distributed random variable and $\mu$ is the linear attenuation coefficient.

Photon transport through the voxel grid is handled using the Amanatides–Woo voxel traversal algorithm (1987). The exponential interaction distance is evaluated locally within each voxel, assuming uniform material properties.

### Local KERMA approximation

The energy transferred to secondary electrons is deposited locally in the interaction voxel, using the approximation:

$$
D \approx K
$$
where $D$ is absorbed dose and $K$ is KERMA.

Secondary charged-particle transport is not explicitly simulated. This approximation is used for megavoltage photon beams in water with 3 mm voxels. Its validity depends on the local charged-particle equilibrium conditions and is therefore treated as a modeling assumption rather than a general clinical accuracy guarantee.

### Radiation spectrum

The simulated beam is polychromatic, with 24 energy bins ranging from 0.25 MeV to 6.00 MeV.

The relative fluence distribution is based on Sheikh-Bagheri and Rogers (2002), with a mean photon energy of approximately 1.96 MeV.

## HPC architecture and GPU optimization

### Sequential CPU baseline

The sequential C++ implementation provides the reference physics model and baseline for the subsequent parallel implementations.

It is used to compare the numerical consistency of the CPU and GPU versions.

### Parallel CPU — OpenMP

The main photon transport loop is parallelized using OpenMP.

Photons are distributed among the available CPU threads using a round-robin scheduling strategy. This implementation provides a multicore baseline for comparison with the GPU versions.

### GPU V1 — Baseline CUDA implementation

The first GPU implementation uses a straightforward photon-parallel strategy:

- One CUDA thread per simulated photon.
- Global-memory phantom storage, with coalesced accesses along the z-axis.
- Constant memory for physical constants and interaction cross sections.
- cuRAND Philox 4×32 random number generation, with independent per-thread states.
- Double-precision dose accumulation using `atomicAdd`.

The main limitations are the variable execution time of individual photon histories and the cost of random number generator initialization.

### GPU V2 — Persistent threads and work stealing

GPU V2 introduces a persistent thread pool and an atomic work queue to improve load balancing.

- A fixed pool of 262,144 persistent CUDA threads.
- Each thread retrieves the next available photon using an atomic increment on a global counter.
- Threads that finish their assigned photon immediately acquire additional work.
- The implementation targets the NVIDIA Tesla T4 GPU, with 40 Streaming Multiprocessors.

This approach reduces the impact of variable photon history lengths and avoids repeatedly creating and destroying threads during the simulation.

### GPU V3 — Selective precision and texture memory

GPU V3 combines numerical precision optimization with improved memory access.

**Selective FP32 arithmetic**

Non-critical operations are converted from double precision to single precision, taking advantage of the higher FP32 throughput of the Tesla T4.

The following quantities retain double precision:

- Photon position coordinates.
- Photon energy.
- Dose accumulation.

The reported theoretical peak throughput of the Tesla T4 is approximately 8.1 TFLOPS for FP32 and 0.25 TFLOPS for FP64.

**3D texture memory**

The voxelized phantom is accessed through CUDA texture memory, which is designed to efficiently handle spatially localized access patterns.

The optimization increases the reported L1 cache hit rate from 29–34% to 91%.

The numerical comparison with the previous implementations shows a difference in fewer than 0.0004% of voxels with non-zero dose.

## Validation and testing

The simulator includes a suite of 10 physical and numerical validation tests, implemented in a Jupyter notebook and applicable to all five versions.

| Test | Validation objective |
|---|---|
| 1 | Effective linear attenuation coefficient estimated from the PDD curve |
| 2 | Beer–Lambert attenuation law in a simplified configuration, with a ±2% tolerance |
| 3 | Consistency of NIST interaction cross sections at 2 MeV in water |
| 4 | Compton scattering angle distribution compared with the theoretical Klein–Nishina mean |
| 5 | Global energy balance and fraction of energy deposited within the phantom |
| 6 | PDD curve shape and reference dose ratios at 10 cm and 20 cm depth |
| 7 | Dose response in a water–bone heterogeneous phantom |
| 8 | Lateral dose profile, penumbra, and Compton scattering |
| 9 | Gamma Index comparison against DOSXYZnrc reference curves |
| 10 | Statistical convergence and uncertainty reduction with increasing photon histories |

These tests assess the consistency of the implemented physical model and the behavior of the numerical simulation. They do not constitute clinical commissioning or certification for treatment planning.

## Output

Each implementation produces:

- CSV files containing the Percentage Depth Dose (PDD) curve along the beam axis.
- CSV files containing the lateral dose profile at 10 cm depth.
- A binary file containing the complete 3D dose grid.
- Performance logs with throughput, kernel execution time, and host-to-device / device-to-host transfer metrics.

## Repository structure

| Directory / File | Description |
|---|---|
| `CPU_Sequenziale/` | Sequential C++ implementation |
| `CPU_Parallelo/` | OpenMP implementation |
| `GPU_V1/` | Baseline CUDA implementation |
| `GPU_V2/` | Persistent threads and work queue |
| `GPU_V3/` | Selective precision and texture memory |
| `Moduli_CPU/` | Shared CPU modules |
| `Moduli_GPU/` | Shared GPU modules |
| `Risultati/` | Simulation results |
| `logs/` | Performance logs |
| `Implementazione progetto.ipynb` | Implementation and validation notebook |
| `Relazione progetto.pdf` | Technical report |
| `Presentazione progetto.pdf` | Project presentation |

## Technologies

`C++` `CUDA C` `OpenMP` `cuRAND` `NVIDIA Tesla T4` `Jupyter Notebook` `Monte Carlo Simulation` `High Performance Computing`
