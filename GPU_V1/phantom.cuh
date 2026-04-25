
#pragma once

#include <cstring>
#include <cstdio>
#include "physics.cuh"

// Phantom Omogeneo (solo acqua) - CPU
inline void build_phantom_water(int *phantom) {
    int total = NX * NY * NZ;
    for (int i = 0; i < total; i++)
        phantom[i] = MAT_WATER;
}

// Phantom acqua + inserto osseo - CPU
inline void build_phantom_hetero(int *phantom) {
    build_phantom_water(phantom);

    int cx = NX / 2;
    int cy = NY / 2;
    int cz = NZ / 2;
    int meta = (int)(2.5 / VOXEL_CM + 0.5);

    int count = 0;
    for (int iz = cz - meta; iz < cz + meta; iz++)
    for (int iy = cy - meta; iy < cy + meta; iy++)
    for (int ix = cx - meta; ix < cx + meta; ix++) {
        if (ix >= 0 && ix < NX && iy >= 0 && iy < NY && iz >= 0 && iz < NZ) {
            phantom[phantom_idx(ix, iy, iz)] = MAT_BONE;
            count++;
        }
    }

    double vol = count * VOXEL_CM * VOXEL_CM * VOXEL_CM;
    printf("Inserto osseo: %d voxel = %.1f cm³  (volume teorico 125 cm³)\n", count, vol);
}

inline void init_dose(double *dose) {
    memset(dose, 0, NX * NY * NZ * sizeof(double));
}
