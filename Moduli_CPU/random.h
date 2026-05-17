
#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>

struct Xoshiro256 {
    uint64_t stato[4];
    // Inizializzazione con un seed a 64 bit usando splitmix64
    explicit Xoshiro256(uint64_t seed) { // con explicit si evitano conversioni automatiche
        auto SplitMix64 = [](uint64_t &stato_corrente) -> uint64_t {
            stato_corrente += 0x9e3779b97f4a7c15ULL;
            uint64_t valore_temporaneo = stato_corrente;
            valore_temporaneo = (valore_temporaneo ^ (valore_temporaneo >> 30)) * 0xbf58476d1ce4e5b9ULL;
            valore_temporaneo = (valore_temporaneo ^ (valore_temporaneo >> 27)) * 0x94d049bb133111ebULL;
            return valore_temporaneo ^ (valore_temporaneo >> 31);
        };
        stato[0] = SplitMix64(seed);
        stato[1] = SplitMix64(seed);
        stato[2] = SplitMix64(seed);
        stato[3] = SplitMix64(seed);
    }

    // Genera un uint64_t casuale
    uint64_t genera_prossimo_numero() {
        const uint64_t numero_estratto = rotazione_sinistra(stato[1] * 5, 7) * 9;
        const uint64_t valore_temporaneo = stato[1] << 17;
        stato[2] ^= stato[0]; stato[3] ^= stato[1]; // ^ indica opertaroe XOR bit a bit
        stato[1] ^= stato[2]; stato[0] ^= stato[3];
        stato[2] ^= valore_temporaneo;
        stato[3] = rotazione_sinistra(stato[3], 45);
        return numero_estratto;
    }

    static uint64_t rotazione_sinistra(const uint64_t valore, int posizioni) {
        return (valore << posizioni) | (valore >> (64 - posizioni));
    }

    double operator()() {
        double numero_decimale;
        do {
            // usa i 53 bit superiori
            numero_decimale = (double)(genera_prossimo_numero() >> 11) * (1.0 / (double)(1ULL << 53)); // ULL intero a 64 bit senza segno
        } while (numero_decimale <= 0.0);
        return numero_decimale;
    }
};

// CDF precalcolata all'inizializzazione
struct Spettro {
    double cdf[NUMERO_BINS_SPETTRO];
    double energie[NUMERO_BINS_SPETTRO];
    double larghezza_intervallo_energetico_bin;

    Spettro() {
        double somma_fluenza = 0.0;
        for (int i = 0; i < NUMERO_BINS_SPETTRO; i++) {
          somma_fluenza += FLUENZA_SPETTRO[i];
        }

        cdf[0] = FLUENZA_SPETTRO[0] / somma_fluenza;
        for (int i = 1; i < NUMERO_BINS_SPETTRO; i++)
            cdf[i] = cdf[i-1] + FLUENZA_SPETTRO[i]/somma_fluenza;
        cdf[NUMERO_BINS_SPETTRO-1] = 1.0;

        for (int i = 0; i < NUMERO_BINS_SPETTRO; i++)
            energie[i] = ENERGIE_SPETTRO[i];

        larghezza_intervallo_energetico_bin = 0.25;
    }

    // Campiona energia con binary search sulla CDF
    double campiona_energia(Xoshiro256 &rng) const {
        double xi = rng();
        // Ricerca binaria sulla CDF
        int indice_limite_inferiore = 0;
        int indice_limite_superiore = NUMERO_BINS_SPETTRO - 1;
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
