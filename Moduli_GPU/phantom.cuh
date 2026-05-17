
#pragma once

#include <cstring>
#include <cstdio>

#ifdef USA_FLOAT
    #include "physics_float.cuh"
#else
    #include "physics.cuh"
#endif

inline void costruzione_phantom_acqua(int *phantom) {
    int capienza_totale = NX * NY * NZ;
    for (int i = 0; i < capienza_totale; i++)
        phantom[i] = MATERIALE_ACQUA;
}

inline void costruzione_phantom_acquaosso(int *phantom) {
    costruzione_phantom_acqua(phantom);

    int cx = NX / 2;
    int cy = NY / 2;
    int cz = NZ / 2;
    int meta_lato_inserto = (int)(2.5 / VOXEL_CM + 0.5);

    for (int iz = cz - meta_lato_inserto; iz < cz + meta_lato_inserto; iz++){
      for (int iy = cy - meta_lato_inserto; iy < cy + meta_lato_inserto; iy++){
        for (int ix = cx - meta_lato_inserto; ix < cx + meta_lato_inserto; ix++) {
            if (ix >= 0 && ix < NX && iy >= 0 && iy < NY && iz >= 0 && iz < NZ) {
                phantom[phantom_idx(ix, iy, iz)] = MATERIALE_OSSO;
            }
        }
      }
    }
}
