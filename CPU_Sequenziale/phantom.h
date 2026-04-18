
#pragma once

#include <cstring>
#include <cstdio>
#include "physics.h"

// Phantom Omogeneo (solo acqua)
inline void build_phantom_water(int *phantom) {
    int total = NX * NY * NZ;
    for (int i = 0; i < total; i++)
        phantom[i] = MAT_WATER;
}

// Phantom acqua + inserto osseo
inline void build_phantom_hetero(int *phantom) {
    build_phantom_water(phantom);

    // Centro del phantom in indici voxel
    int cx = NX / 2;
    int cy = NY / 2;
    int cz = NZ / 2;

    int meta_lato_inserto = (int)(2.5 / VOXEL_CM + 0.5);

    int count = 0;
    for (int iz = cz - meta_lato_inserto; iz < cz + meta_lato_inserto; iz++)
    for (int iy = cy - meta_lato_inserto; iy < cy + meta_lato_inserto; iy++)
    for (int ix = cx - meta_lato_inserto; ix < cx + meta_lato_inserto; ix++) {
        if (ix >= 0 && ix < NX && iy >= 0 && iy < NY && iz >= 0 && iz < NZ) { // controlli per non scrivere fuori dalla memoria del phantom
            phantom[phantom_idx(ix, iy, iz)] = MAT_BONE;
            count++;
        }
    }

    double volume_fisico_totale = count * VOXEL_CM * VOXEL_CM * VOXEL_CM;
    printf("Inserto osseo: %d voxel = %.1f cm³  ( volume teorico 125 cm³)\n", count, volume_fisico_totale);
}

// Inizializzazione dose grid a zero
inline void init_dose(double *dose) {
    memset(dose, 0, NX * NY * NZ * sizeof(double));
}
