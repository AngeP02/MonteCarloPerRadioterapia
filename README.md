# Monte Carlo per Radioterapia in CUDA
Simulatore Monte Carlo per il calcolo della dose in radioterapia, implementato su architetture CPU e GPU (CUDA), sviluppato nell'ambito del corso di **High Performance Computing**.


## Panoramica

Questo progetto implementa un simulatore Monte Carlo per la **distribuzione della dose in radioterapia oncologica**, con simulazione del trasporto di fotoni in phantom voxelizzati omogenei ed eterogenei. Il sistema riproduce la fisica del fascio clinico da **6 MV** generato da un acceleratore lineare LINAC (Varian 2100C), conforme ai dati spettrali di Sheikh-Bagheri & Rogers.

Il metodo Monte Carlo è lo standard di riferimento in dosimetria computazionale grazie alla sua capacità di modellare stocasticamente le singole interazioni fotone–materia all'interno di griglie voxelizzate derivate da immagini TAC. Per raggiungere una precisione clinicamente accettabile (incertezza ≤ 5%) sono necessari ordini di grandezza di 10⁷–10⁸ fotoni simulati, il che rende indispensabile l'accelerazione HPC.

L'obiettivo principale è lo sviluppo e il confronto comparativo di cinque implementazioni su architetture differenti:

| Versione | Tecnologia | Parallelismo | Ottimizzazioni principali |
|---|---|---|---|
| CPU Sequenziale | C++ | Nessuno | Baseline fisica validata |
| CPU Parallela | C++ + OpenMP | Multi-thread, Round Robin | Distribuzione del carico tra core |
| GPU V1 | CUDA C | 1 thread per fotone | Constant memory per sezioni d'urto, cuRAND Philox |
| GPU V2 | CUDA C | Persistent thread pool | Work stealing con coda atomica |
| GPU V3 | CUDA C | Persistent thread pool | Conversione FP32 selettiva, texture memory |

---

### Interazioni fotone–materia

La simulazione implementa i tre processi fondamentali nell'intervallo energetico della radioterapia (0.01–20 MeV), con sezioni d'urto derivate dal database **NIST XCOM**:

- **Effetto fotoelettrico** — assorbimento totale del fotone; l'energia viene depositata localmente nel voxel corrente
- **Diffusione Compton** — campionamento angolare tramite il **metodo di Kahn** (rejection sampling sulla distribuzione di Klein–Nishina); il fotone continua con nuova direzione ed energia ridotta; l'energia ceduta è depositata localmente
- **Produzione di coppie** — generazione di due fotoni di annichilazione da 0.511 MeV; l'energia cinetica residua viene depositata localmente

### Trasporto e campionamento

Il cammino libero medio è campionato tramite inversione della **legge di Beer–Lambert**:

```
λ = -ln(ξ) / μ
```

Il percorso attraverso la griglia è gestito tramite l'algoritmo di **voxel traversal** (Amanatides & Woo, 1987), che applica la legge esponenziale localmente all'interno di ciascun voxel dove il materiale è considerato uniforme.

### Approssimazione KERMA locale

L'energia trasferita dagli elettroni secondari viene depositata nel voxel di interazione (**dose ≈ KERMA**), senza simulare il trasporto esplicito delle particelle cariche. Questa approssimazione è fisicamente giustificata per fasci di megavoltaggio in acqua con voxel da 3 mm, in quanto il range degli elettroni Compton è dell'ordine del centimetro.

### Spettro energetico

Viene simulato un fascio policromatico con 24 bin da 0.25 MeV a 6.00 MeV, con fluenza relativa da Sheikh-Bagheri & Rogers (2002). L'energia media dello spettro è ~1.96 MeV.

---

## Architettura del progetto

#### Versione CPU Sequenziale

Scritta in C++, funge da baseline fisica e da riferimento di validazione per tutte le versioni successive.

#### Versione CPU Parallela (OpenMP)

Parallelizzazione del loop principale tramite **OpenMP con distribuzione Round Robin** dei fotoni tra i thread disponibili. La versione rimane limitata dal numero di core fisici della macchina host.

#### Versioni GPU (CUDA)

**GPU V1 — Versione base**
- Un thread CUDA per fotone simulato
- Phantom in memoria globale con accesso coalescente lungo l'asse z
- Costanti fisiche e sezioni d'urto in **constant memory** (cache broadcast)
- Generatore **cuRAND Philox 4×32** con stato indipendente per thread
- Accumulo della dose con `atomicAdd` su `double`
- Criticità: sbilanciamento del carico (durata stocastica variabile), costo di inizializzazione cuRAND

**GPU V2 — Work Stealing**
- Pool fisso di `1024 × 256 = 262.144 thread` persistenti per tutta la durata del kernel
- **Work queue atomica**: ogni thread acquisisce il prossimo fotone disponibile con `atomicAdd` su contatore globale; thread veloci non attendono ma riprendono immediatamente lavoro
- Saturazione ottimale degli Streaming Multiprocessor della Tesla T4 (40 SM × 16 blocchi attivi)
- Eliminazione del costo di creazione/distruzione dinamica dei thread

**GPU V3 — Precision + Texture Memory**
- **Conversione selettiva FP32**: operazioni non critiche portate da double a float, sfruttando il throughput FP32 (8.1 TFLOPS vs 0.25 TFLOPS FP64 su Tesla T4)
- Coordinate di posizione, energia del fotone e accumulo della dose mantengono doppia precisione
- **Texture memory** per il phantom voxelizzato: ottimizza l'accesso 3D stocastico, portando la cache L1 hit dal 29–34% al 91%
- Differenza fisica rispetto alle versioni precedenti: < 0.0004% dei voxel con dose > 0 (fisicamente accettabile)

---

### Output
Ciascuna versione produce:
- File CSV con la curva **PDD** (Percent Depth Dose) lungo l'asse del fascio
- File CSV con il **profilo laterale** della dose a 10 cm di profondità
- File binario con la **griglia 3D** completa della dose
- File di **log** con metriche di performance (throughput, tempo kernel, trasferimenti H↔D)

---

## Validazione fisica

Il simulatore è stato sottoposto a una suite di **10 test di validazione** fisica e clinica, eseguibili tramite il notebook Jupyter su tutte le versioni (cambiando solo le costanti dei file di output):

| Test | Verifica |
|---|---|
| Test 1 | Coefficiente di attenuazione lineare effettivo (regressione su PDD in zona 5–20 cm, range atteso: 0.030–0.055 cm⁻¹) |
| Test 2 | Legge di Beer–Lambert pura (versione semplificata, tolleranza ±2% su 7 profondità di riferimento) |
| Test 3 | Coerenza sezioni d'urto NIST: a 2 MeV in acqua, Compton > 99.9%, fotoelettrico < 0.01%, coppie = 0 |
| Test 4 | Validazione stocastica dell'angolo di scattering Compton vs. coseno medio teorico di Klein–Nishina (5 energie, errore < 5%) |
| Test 5 | Bilancio energetico globale: frazione depositata nel phantom tra 30% e 90% |
| Test 6 | Forma del PDD clinico: picco in 0–2 cm, D(10cm)/D_max ∈ [0.60, 0.82], D(20cm)/D(10cm) ∈ [0.45, 0.72] |
| Test 7 | Eterogeneità acqua–osso: incremento della dose nel voxel osseo e attenuazione a valle |
| Test 8 | Penombra laterale e diffusione Compton |
| Test 9 | **Gamma Index** vs. curve di riferimento DOSXYZnrc (Sheikh-Bagheri & Rogers) |
| Test 10 | Convergenza statistica: riduzione dell'incertezza con √N fotoni |
