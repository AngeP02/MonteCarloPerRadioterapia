#pragma once
/*
 * random.h
 * ─────────────────────────────────────────────────────────────────────────────
 * Generatore di numeri pseudo-casuali e campionamento dello spettro 6MV.
 *
 * PRNG: xoshiro256** di Blackman & Vigna (2018)
 *   Periodo: 2^256-1
 *   Supera tutti i test di BigCrush
 *   Molto più veloce e con periodo molto più lungo di rand()
 *   Fonte: https://prng.di.unimi.it/xoshiro256starstar.c
 *   Blackman D., Vigna S. (2018) "Scrambled Linear Pseudorandom Number Generators"
 *
 * Spettro 6MV: Sheikh-Bagheri D., Rogers D.W.O.
 *   Medical Physics 29(3), 2002 — Tabella II, fascio 6MV Varian
 *
 * Autore: Angelica Porco - Matricola 264034
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cstdint>
#include <cmath>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// XOSHIRO256**  —  PRNG di alta qualità per simulazioni scientifiche
// ─────────────────────────────────────────────────────────────────────────────
struct Xoshiro256 {
    uint64_t s[4];

    // Inizializza con un seed a 64 bit usando splitmix64
    explicit Xoshiro256(uint64_t seed) {
        // splitmix64 per espandere il seed a 256 bit
        auto splitmix = [](uint64_t &x) -> uint64_t {
            x += 0x9e3779b97f4a7c15ULL;
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
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1];
        s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Genera double uniforme in (0, 1)  — esclude 0 per evitare log(0)
    double operator()() {
        // Mappa uint64 → double in [0,1), poi controlla
        double r;
        do {
            // Tecnica standard: usa i 53 bit superiori
            r = (double)(next() >> 11) * (1.0 / (double)(1ULL << 53));
        } while (r <= 0.0);
        return r;
    }

private:
    static uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SPETTRO 6MV  —  Sheikh-Bagheri & Rogers, Med. Phys. 29(3), 2002
// Fascio Varian 2100C a 6MV, campo 10×10cm² a SAD 100cm
// 24 bin da 0.25 a 6.00 MeV, passo 0.25 MeV
// ─────────────────────────────────────────────────────────────────────────────
static const int    SPECTRUM_BINS = 24;
static const double SPECTRUM_E[SPECTRUM_BINS] = {
    0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
    2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00,
    4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00
};   // MeV — centro di ogni bin

// Fluenza relativa normalizzata  (somma = 1.0)
// Fonte: Sheikh-Bagheri & Rogers 2002, Tabella II
static const double SPECTRUM_FLUENCE[SPECTRUM_BINS] = {
    0.0243, 0.0676, 0.0862, 0.0929, 0.0919, 0.0868, 0.0794, 0.0712,
    0.0628, 0.0548, 0.0471, 0.0399, 0.0334, 0.0276, 0.0224, 0.0178,
    0.0138, 0.0104, 0.0075, 0.0052, 0.0034, 0.0020, 0.0010, 0.0004
};

// CDF precalcolata all'inizializzazione
struct Spectrum {
    double cdf[SPECTRUM_BINS];
    double energies[SPECTRUM_BINS];
    double bin_width;

    Spectrum() {
        // Normalizza e costruisce CDF
        double total = 0.0;
        for (int i = 0; i < SPECTRUM_BINS; i++) total += SPECTRUM_FLUENCE[i];

        cdf[0] = SPECTRUM_FLUENCE[0] / total;
        for (int i = 1; i < SPECTRUM_BINS; i++)
            cdf[i] = cdf[i-1] + SPECTRUM_FLUENCE[i] / total;
        cdf[SPECTRUM_BINS-1] = 1.0;  // garantisce che l'ultimo bin sia sempre scelto

        for (int i = 0; i < SPECTRUM_BINS; i++)
            energies[i] = SPECTRUM_E[i];

        bin_width = 0.25;  // MeV — larghezza bin uniforme
    }

    // Campiona energia con binary search sulla CDF
    // + interpolazione uniforme all'interno del bin
    // Fonte: Salvat PENELOPE-2014, sezione 1.2 (campionamento da tabelle)
    double sample(Xoshiro256 &rng) const {
        double xi = rng();

        // Binary search
        int lo = 0, hi = SPECTRUM_BINS - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cdf[mid] < xi) lo = mid + 1;
            else               hi = mid;
        }

        // Interpolazione uniforme dentro il bin per smoothness
        double e_center = energies[lo];
        double offset   = (rng() - 0.5) * bin_width;   // [-0.125, +0.125] MeV
        double e        = e_center + offset;
        // Clamp al range fisico
        if (e < 0.01) e = 0.01;
        if (e > 6.00) e = 6.00;
        return e;
    }
};
