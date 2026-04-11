#pragma once
/*
 * phantom.h
 * ─────────────────────────────────────────────────────────────────────────────
 * Costruzione del phantom voxelizzato.
 *
 * Phantom 1: acqua omogenea 30×30×30 cm³ (100³ voxel, 3mm lato)
 *   Standard IAEA per validazione MC-RT
 *   Fonte: IAEA Technical Reports Series No. 430 (2004)
 *
 * Phantom 2: acqua + inserto osseo 5×5×5 cm³ al centro
 *   Per validazione della gestione delle eterogeneità
 *
 * Autore: Angelica Porco - Matricola 264034
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cstring>
#include <cstdio>
#include "physics.h"

// ─────────────────────────────────────────────────────────────────────────────
// PHANTOM OMOGENEO — acqua pura
// ─────────────────────────────────────────────────────────────────────────────
inline void build_phantom_water(int *phantom) {
    int total = NX * NY * NZ;
    for (int i = 0; i < total; i++)
        phantom[i] = MAT_WATER;
}

// ─────────────────────────────────────────────────────────────────────────────
// PHANTOM ETEROGENEO — acqua + inserto osseo centrale 5×5×5 cm³
// ─────────────────────────────────────────────────────────────────────────────
inline void build_phantom_hetero(int *phantom) {
    build_phantom_water(phantom);

    // Centro del phantom in indici voxel
    int cx = NX / 2;   // 50
    int cy = NY / 2;   // 50
    int cz = NZ / 2;   // 50

    // Metà lato inserto: 2.5 cm / 0.3 cm/voxel ≈ 8 voxel
    int half = (int)(2.5 / VOXEL_CM + 0.5);   // 8

    int count = 0;
    for (int iz = cz - half; iz < cz + half; iz++)
    for (int iy = cy - half; iy < cy + half; iy++)
    for (int ix = cx - half; ix < cx + half; ix++) {
        if (ix >= 0 && ix < NX && iy >= 0 && iy < NY && iz >= 0 && iz < NZ) {
            phantom[phantom_idx(ix, iy, iz)] = MAT_BONE;
            count++;
        }
    }

    double vol_bone = count * VOXEL_CM * VOXEL_CM * VOXEL_CM;
    printf("  Inserto osseo: %d voxel = %.1f cm³  (target 125 cm³)\n",
           count, vol_bone);
}

// ─────────────────────────────────────────────────────────────────────────────
// INIZIALIZZA DOSE GRID A ZERO
// ─────────────────────────────────────────────────────────────────────────────
inline void init_dose(double *dose) {
    memset(dose, 0, NX * NY * NZ * sizeof(double));
}
