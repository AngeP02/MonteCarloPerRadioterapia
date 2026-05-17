
#pragma once

#include <cmath>
#include <cassert>

// COSTANTI FISIHCHE
static const double ME_C2    = 0.51099895;  // MeV  massa a riposo elettrone
static const double PI       = 3.14159265358979323846;
static const double ECUT     = 0.010;       // MeV  cutoff fotoni  (10 keV)
static const double PCUT     = 0.100;       // MeV  cutoff elettroni

// GEOMETRIA PHANTOM
static const int    NX = 100, NY = 100, NZ = 100;   // voxel per asse
static const double VOXEL_CM = 0.30;                // lato voxel [cm] = 3 mm
static const double PHANTOM_CM = NX * VOXEL_CM;     // 30.0 cm per asse
static const double SEMI_AMPIEZZA_CAMPO = 5.0;

// INDICI MATERIALI
#define MATERIALE_ACQUA 0   // acqua  ρ = 1.000 g/cm^3
#define MATERIALE_OSSO 1   // osso (ICRU)  ρ = 1.850 g/cm^3
#define NUMERO_MATERIALI 2   // numero materiali disponibili

// DENSITÀ [g/cm^3]
static const double DENSITA[NUMERO_MATERIALI] = { 1.000, 1.850 };

// GRIGLIA ENERGETICA [MeV]  (28 punti, da 0.01 a 20 MeV)
static const int PUNTI_CAMPIONAMENTO = 28;
static const double GRIGLIA_ENERGIA[PUNTI_CAMPIONAMENTO] = {
    0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100,
    0.150, 0.200, 0.300, 0.400, 0.500, 0.600, 0.800, 1.000, 1.250,
    1.500, 2.000, 3.000, 4.000, 5.000, 6.000, 8.000, 10.000,
    15.000, 20.000
};

// COEFFICIENTI μ/ρ [cm^2/g]  per ogni materiale e processo -> [materiale][bin_energia]
static const double PROBABILITA_TOTALE[NUMERO_MATERIALI][PUNTI_CAMPIONAMENTO] = {
    // ACQUA
    { 5.329, 1.673, 0.8096, 0.3756, 0.2683, 0.2269, 0.2059, 0.1837, 0.1707,
      0.1505, 0.1370, 0.1186, 0.1061, 0.09687, 0.09007, 0.07865, 0.07072, 0.06323,
      0.05754, 0.04942, 0.03969, 0.03403, 0.03031, 0.02770, 0.02429, 0.02219,
      0.01941, 0.01813 },
    // OSSO
    { 19.89, 7.131, 3.085, 1.012, 0.5475, 0.3941, 0.3178, 0.2595, 0.2368,
      0.1958, 0.1698, 0.1393, 0.1222, 0.1107, 0.1018, 0.08795, 0.07838, 0.06945,
      0.06283, 0.05351, 0.04257, 0.03624, 0.03209, 0.02913, 0.02536, 0.02296,
      0.01978, 0.01832 }
};

// EFFETTO FOTOELETTRICO
static const double PROBABILITA_FOTOELETTRICO[NUMERO_MATERIALI][PUNTI_CAMPIONAMENTO] = {
    // ACQUA
    { 4.944, 1.374, 0.5195, 0.1036, 0.02407, 0.005800, 0.001334, 5.510e-5, 3.998e-5,
      2.799e-6, 2.200e-7, 1.400e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    // OSSO
    { 19.35, 6.833, 2.818, 0.7469, 0.2837, 0.1152, 0.04660, 0.008680, 0.001900,
      1.800e-4, 2.000e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
};

// SCATTERING COMPTON
static const double PROBABILITA_COMPTON[NUMERO_MATERIALI][PUNTI_CAMPIONAMENTO] = {
    // ACQUA
    { 0.3854, 0.2988, 0.2672, 0.2651, 0.2595, 0.2476, 0.2329, 0.1984, 0.1661,
      0.1505, 0.1370, 0.1186, 0.1061, 0.09687, 0.09007, 0.07865, 0.07072, 0.06323,
      0.05754, 0.04942, 0.03969, 0.03403, 0.03031, 0.02770, 0.02429, 0.02219,
      0.01878, 0.01719 },
    // OSSO
    { 0.4869, 0.2684, 0.2503, 0.2465, 0.2429, 0.2310, 0.2172, 0.1848, 0.1548,
      0.1400, 0.1275, 0.1103, 0.09870, 0.09010, 0.08377, 0.07313, 0.06575, 0.05862,
      0.05338, 0.04579, 0.03667, 0.03133, 0.02784, 0.02539, 0.02217, 0.02016,
      0.01702, 0.01552 }
};

// PRODUZIONE DI COPPIE
static const double PROBABILITA_COPPIA[NUMERO_MATERIALI][PUNTI_CAMPIONAMENTO] = {
    // ACQUA
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.000630, 0.000940 },
    // OSSO
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.002760, 0.002800 }
};

// SPETTRO 6MV
static const int NUMERO_BINS_SPETTRO = 24;
static const double ENERGIE_SPETTRO[NUMERO_BINS_SPETTRO] = {
    0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
    2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00,
    4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00
};

// Fluenza relativa normalizzata  (somma = 1.0)
static const double FLUENZA_SPETTRO[NUMERO_BINS_SPETTRO] = {
    0.0243, 0.0676, 0.0862, 0.0929, 0.0919, 0.0868, 0.0794, 0.0712,
    0.0628, 0.0548, 0.0471, 0.0399, 0.0334, 0.0276, 0.0224, 0.0178,
    0.0138, 0.0104, 0.0075, 0.0052, 0.0034, 0.0020, 0.0010, 0.0004
};

// INTERPOLAZIONE LINEARE SU GRIGLIA ENERGETICA
inline double interpolazione_lineare_energia(double energia_mev, const double tabella[PUNTI_CAMPIONAMENTO]) {
    if (energia_mev <= GRIGLIA_ENERGIA[0])
        return tabella[0];
    if (energia_mev >= GRIGLIA_ENERGIA[PUNTI_CAMPIONAMENTO-1])
        return tabella[PUNTI_CAMPIONAMENTO-1];

    int indice_limite_inferiore = 0;
    int indice_imite_superiore = PUNTI_CAMPIONAMENTO - 1;

    while (indice_imite_superiore - indice_limite_inferiore > 1) {
        int punto_centrale = (indice_limite_inferiore + indice_imite_superiore) / 2;
        if (GRIGLIA_ENERGIA[punto_centrale] <= energia_mev){
            indice_limite_inferiore = punto_centrale;
        }else{
            indice_imite_superiore = punto_centrale;
        }
    }

    double fattore_interpolazione = (energia_mev - GRIGLIA_ENERGIA[indice_limite_inferiore]) / (GRIGLIA_ENERGIA[indice_imite_superiore] - GRIGLIA_ENERGIA[indice_limite_inferiore]);
    return tabella[indice_limite_inferiore] * (1.0 - fattore_interpolazione) + tabella[indice_imite_superiore] * fattore_interpolazione;
}

// CALCOLO MU TOTALE
inline double calcolo_attenuazione_totale(double energia, int materiale) {
    return interpolazione_lineare_energia(energia, PROBABILITA_TOTALE[materiale]) * DENSITA[materiale];
}
inline double calcolo_attenuazione_fotoelettrico(double energia, int materiale) {
    return interpolazione_lineare_energia(energia, PROBABILITA_FOTOELETTRICO[materiale]) * DENSITA[materiale];
}
inline double calcolo_attenuazione_compton(double energia, int materiale) {
    return interpolazione_lineare_energia(energia, PROBABILITA_COMPTON[materiale]) * DENSITA[materiale];
}
inline double calcolo_attenuazione_coppie(double energia, int materiale) {
    return interpolazione_lineare_energia(energia, PROBABILITA_COPPIA[materiale]) * DENSITA[materiale];
}

// SELEZIONE TIPO DI INTERAZIONE
// Restituisce: 0=fotoelettrico, 1=Compton, 2=produzione coppie
// xi: numero casuale uniforme in [0,1)
inline int seleziona_tipo_interazione(double energia, int materiale, double xi) {
    double probabilita_totale = calcolo_attenuazione_totale(energia, materiale);

    if (probabilita_totale <= 0.0)
        return 1;

    double probabilita_fotoelettrico = calcolo_attenuazione_fotoelettrico(energia, materiale) / probabilita_totale;
    double probabilita_compton = calcolo_attenuazione_compton(energia, materiale) / probabilita_totale;

    if (xi < probabilita_fotoelettrico)
        return 0;   // fotoelettrico
    if (xi < probabilita_fotoelettrico + probabilita_compton)
        return 1; // compton
    return 2;  // produzione di coppie
}

// INDICE LINEARE PHANTOM CON ROW MAJOR ORDER
inline int phantom_idx(int ix, int iy, int iz) {
    return ix + NX * (iy + NY * iz);
}

// CONTROLLO COORDINATE CONTORNO CUBO
inline bool verifica_confini(double x, double y, double z) {
    return x >= 0.0 && x < PHANTOM_CM &&
           y >= 0.0 && y < PHANTOM_CM &&
           z >= 0.0 && z < PHANTOM_CM;
}

// CONTROLLO PSIZIONE IN VOXEL
inline int vox(double coord) {
    int indice_voxel = (int)(coord / VOXEL_CM);
    if (indice_voxel < 0)
        indice_voxel = 0;
    if (indice_voxel >= NX)
        indice_voxel = NX - 1;
    return indice_voxel;
}
