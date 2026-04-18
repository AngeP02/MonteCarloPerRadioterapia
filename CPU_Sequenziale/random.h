
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>

// XOSHIRO256
struct Xoshiro256 {
    uint64_t s[4];

    // Inizializzazione con un seed a 64 bit usando splitmix64
    explicit Xoshiro256(uint64_t seed) { // con explicit si evitano conversioni automatiche
        auto splitmix = [](uint64_t &x) -> uint64_t {
            x += 0x9e3779b97f4a7c15ULL; // rappresentazione a 64 bit della sezione aurea (2^64/phi)
            uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31);
        };
        s[0] = splitmix(seed);
        s[1] = splitmix(seed);
        s[2] = splitmix(seed);
        s[3] = splitmix(seed);
    }

    // Genera un uint64_t casuale
    uint64_t next() {
        const uint64_t result = rotate_left(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; // ^ indica opertaroe XOR bit a bit
        s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotate_left(s[3], 45);
        return result;
    }

    // Genera double uniforme in (0, 1) escludendo 0 per evitare log(0)
    double operator()() {
        double r;
        do {
            // usa i 53 bit superiori
            r = (double)(next() >> 11) * (1.0 / (double)(1ULL << 53)); // ULL intero a 64 bit senza segno
        } while (r <= 0.0);
        return r;
    }

private:
    static uint64_t rotate_left(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

// SPETTRO 6MV
static const int SPECTRUM_BINS = 24;
static const double SPECTRUM_E[SPECTRUM_BINS] = {
    0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
    2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00,
    4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00
};

// Fluenza relativa normalizzata  (somma = 1.0)
static const double SPECTRUM_FLUENCE[SPECTRUM_BINS] = {
    0.0243, 0.0676, 0.0862, 0.0929, 0.0919, 0.0868, 0.0794, 0.0712,
    0.0628, 0.0548, 0.0471, 0.0399, 0.0334, 0.0276, 0.0224, 0.0178,
    0.0138, 0.0104, 0.0075, 0.0052, 0.0034, 0.0020, 0.0010, 0.0004
};

// CDF precalcolata all'inizializzazione
struct Spectrum {
    double cdf[SPECTRUM_BINS];
    double energie[SPECTRUM_BINS];
    double larghezza_intervallo_energetico_bin;

    Spectrum() {
        double somma_fluence = 0.0;
        for (int i = 0; i < SPECTRUM_BINS; i++) {
          somma_fluence += SPECTRUM_FLUENCE[i];
        }

        cdf[0] = SPECTRUM_FLUENCE[0] / somma_fluence;
        for (int i = 1; i < SPECTRUM_BINS; i++)
            cdf[i] = cdf[i-1] + SPECTRUM_FLUENCE[i] / somma_fluence;
        cdf[SPECTRUM_BINS-1] = 1.0;

        for (int i = 0; i < SPECTRUM_BINS; i++)
            energie[i] = SPECTRUM_E[i];

        larghezza_intervallo_energetico_bin = 0.25;
    }

    // Campiona energia con binary search sulla CDF
    double sample(Xoshiro256 &rng) const {
        double xi = rng();
        // Ricerca binaria sulla CDF
        int indice_limite_inferiore = 0;
        int indice_limite_superiore = SPECTRUM_BINS - 1;
        while (indice_limite_inferiore < indice_limite_superiore) {
            int punto_centrale = (indice_limite_inferiore + indice_limite_superiore) / 2;
            if (cdf[punto_centrale] < xi){
                indice_limite_inferiore = punto_centrale + 1;
            }
            else{
                indice_limite_superiore = punto_centrale;
            }
        }

        double energia_centrale = energie[indice_limite_inferiore];
        double offset = (rng() - 0.5) * larghezza_intervallo_energetico_bin;
        double energia = energia_centrale + offset;

        if (energia < 0.01){
           energia = 0.01;
        }
        if (energia > 6.00){
          energia = 6.00;
        }
        return energia;
    }
};
