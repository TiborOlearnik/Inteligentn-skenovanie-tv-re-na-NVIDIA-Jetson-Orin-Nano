# Inteligentné skenovanie tváre na NVIDIA Jetson Orin Nano

Systém na detekciu a biometrickú autentifikáciu osôb v reálnom čase optimalizovaný pre platformu **NVIDIA Jetson Orin Nano 8GB** (architektúra Ampere, Compute Capability `87`). Projekt využíva hardvérovú GPU akceleráciu (**CUDA + cuDNN**) na detekciu tvárí pomocou konvolučnej neurónovej siete (CNN) a extrakciu 128-rozmerných biometrických vektorov (ResNet).

Knižničný základ a nastavenie závislostí pre detekciu a rozpoznávanie tvárí na platforme Jetson vychádzajú z repozitára [**lovnishverma/FaceRecognition-Attendance**](https://github.com/lovnishverma/FaceRecognition-Attendance).

## Obsah

1. [Architektúra a štruktúra repozitára](#1-architektúra-a-štruktúra-repozitára)
2. [Princíp fungovania (Biometrický pipeline)](#2-princíp-fungovania-biometrický-pipeline)
3. [Požiadavky a príprava systému (Jetson Orin Nano)](#3-požiadavky-a-príprava-systému-jetson-orin-nano)
4. [Inštalácia a kompilácia s podporou CUDA](#4-inštalácia-a-kompilácia-s-podporou-cuda)
5. [Používateľská príručka](#5-používateľská-príručka)
6. [Dokumentácia kľúčových modulov](#6-dokumentácia-kľúčových-modulov)
7. [Riešenie častých problémov](#7-riešenie-častých-problémov)
8. [Zdroje a knižničný základ](#8-zdroje-a-knižničný-základ)

## 1. Architektúra a štruktúra repozitára

```text
.
├── web_app.py                        # Webový server (Flask) na registráciu, overovanie a správu osôb
├── test_photos.py                    # Offline dávkový test autentifikácie nad statickými fotografiami
├── testphotoscas.py                  # Offline test autentifikácie s meraním času a zápisom logu
├── capture.py                        # CLI skript na snímanie viacerých uhlov tváre cez kameru Jetsonu
├── main.py                           # Real-time dochádzkový skript
├── opencv_face_detector_uint8.pb     # Predtrénovaný TensorFlow model pre OpenCV DNN detektor
├── opencv_face_detector.pbtxt        # Konfiguračný graf pre OpenCV DNN detektor tvárí
├── requirements.txt                  # Zoznam základných Python knižníc
├── README.md                         # Dokumentácia projektu a návod na spustenie
└── Attendance_data/                  # Zdieľaná databáza tvárí (vytvorí sa automaticky)
    ├── .encodings_cache.pkl          # Vyrovnávacia pamäť vypočítaných 128-D vektorov tvárí
    ├── web_auth_log.csv              # História pokusov o autentifikáciu (čas, zdroj, stav, vzdialenosť)
    └── <MENO_OSOBY>/                 # Priečinok konkrétneho používateľa (napr. PETER/)
        ├── PETER_1.jpg
        └── PETER_2.jpg
```

## 2. Princíp fungovania (Biometrický pipeline)

Overenie identity prebieha v troch nadväzujúcich krokoch:

1. **Detekcia tváre na GPU (`CNN` / `DNN`):**
   * Vo `web_app.py` sa používa MMOD CNN detektor z knižnice `dlib` (`model="cnn"`), ktorý beží priamo na jadrách GPU cez CUDA a cuDNN.
   * V `capture.py` sa na rýchly živý náhľad pri zbere dát používa OpenCV DNN modul (`opencv_face_detector_uint8.pb`) s CUDA backendom.

2. **Extrakcia príznakov (Face Embedding):**
   * Z nájdeného výrezu tváre vygeneruje hlboká reziduálna sieť (ResNet) vektor 128 reálnych čísel $\mathbf{v} \in \mathbb{R}^{128}$, ktorý reprezentuje geometrické črty tváre.

3. **Porovnanie (Euklidovská vzdialenosť):**
   * Vypočíta sa Euklidovská vzdialenosť $d$ medzi testovanou tvárou $\mathbf{p}$ a uloženými referenčnými vektormi $\mathbf{q}$ v databáze:
     $$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{128} (p_i - q_i)^2}$$
   * Ak je najmenšia nameraná vzdialenosť menšia alebo rovná nastavenému prahu **Tolerance** (predvolene $0.50$), osoba je **AUTORIZOVANÁ**.

| Nameraná vzdialenosť $d$ | Interpretácia zhody |
| :--- | :--- |
| $0.00$ | Identická fotografia |
| $0.25 - 0.40$ | Veľmi silná zhoda (rovnaká osoba, dobré svetelné podmienky) |
| $0.40 - 0.50$ | Spoľahlivá zhoda (rovnaká osoba, mierne odlišný uhol alebo osvetlenie) |
| $> 0.50$ | Odlišná osoba (prístup zamietnutý pri predvolenej tolerancii) |

## 3. Požiadavky a príprava systému (Jetson Orin Nano)

* **Hardvér:** NVIDIA Jetson Orin Nano 8GB (CUDA architektúra `87`)
* **OS / Softvér:** Ubuntu (JetPack 6+), Python 3.12 (alebo 3.10+), predinštalovaný NVIDIA CUDA Toolkit a cuDNN
* **Kamera (voliteľné):** CSI kamera (MIPI) alebo USB webkamera (systém umožňuje testovanie aj čisto zo súborov bez kamery)

### Nastavenie 8 GB Swap pamäte (nutné pred kompiláciou `dlib`)

Kompilácia `dlib` s podporou CUDA vyžaduje viac pamäte než 8 GB zdieľanej RAM. Pred inštaláciou nastav 8 GB swap súbor:

```bash
# Kontrola aktuálneho swapu
swapon --show

# Ak /swapfile už existuje a je menší, najprv ho odpoj
sudo swapoff /swapfile

# Vytvorenie a aktivácia 8 GB swap súboru
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Trvalé pripojenie po reštarte systému
grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

### Inštalácia systémových balíkov

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config \
    libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev \
    libpng-dev libjpeg-dev libwebp-dev python3-venv python3-dev
```

## 4. Inštalácia a kompilácia s podporou CUDA

> **Dôležité upozornenie:** Knižnicu `dlib` je nutné skompilovať zo zdrojového kódu **pred** inštaláciou `face_recognition`. Ak by sa nainštalovala priamo cez `pip`, stiahla by sa verzia bez podpory grafickej karty (CPU-only).

### Krok 1: Klonovanie repozitára a vytvorenie virtuálneho prostredia

```bash
git clone <URL_TVOJHO_GITHUB_REPOZITARA>
cd <NAZOV_PRIECINKA_REPOZITARA>

python3 -m venv ~/dlib_env
source ~/dlib_env/bin/activate
```

### Krok 2: Inštalácia kompatibilných verzií základných balíkov

Pre Python 3.12 na architektúre `aarch64` je potrebné nainštalovať `numpy` radu `1.26.x`, `opencv-python` radu `4.x` a `setuptools<80` (kvôli zachovaniu modulu `pkg_resources` pre `face_recognition_models`):

```bash
pip install --upgrade pip wheel
pip install "setuptools<80" "numpy==1.26.4" "opencv-python<5" flask pytz
```

### Krok 3: Kompilácia `dlib` s CUDA pre Jetson Orin (`arch=87`)

Zdrojový kód `dlib` naklonuj mimo priečinka projektu (napr. do `~/dlib`), aby nezaťažoval tvoj repozitár:

```bash
cd ~
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir -p build && cd build

# Konfigurácia CMake pre Jetson Orin (CUDA architektúra 87, bez Intel AVX inštrukcií)
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=0 -DCMAKE_CUDA_ARCHITECTURES=87

# Kompilácia C++ jadra na všetkých jadrách procesora (trvá cca 5 - 10 minút)
cmake --build . --config Release -j$(nproc)

# Inštalácia Python modulu do aktívneho prostredia dlib_env
cd ..
python3 setup.py install --yes DLIB_USE_CUDA --set CMAKE_CUDA_ARCHITECTURES=87
```

### Krok 4: Inštalácia `face_recognition` a overenie GPU

```bash
pip install face_recognition git+https://github.com/ageitgey/face_recognition_models

# Overenie, že dlib vidí GPU a CUDA je aktívna (musí vypísať True a 1)
python3 -c "import dlib, face_recognition; print('CUDA:', dlib.DLIB_USE_CUDA, '| GPU:', dlib.cuda.get_num_devices())"
```

## 5. Používateľská príručka

### A. Spustenie webovej aplikácie (`web_app.py`)

1. Aktivuj prostredie a spusti server v priečinku repozitára:
   ```bash
   source ~/dlib_env/bin/activate
   cd <NAZOV_PRIECINKA_REPOZITARA>
   python3 web_app.py
   ```
2. Otvor rozhranie v prehliadači:
   * **Lokálne na Jetsone:** `http://localhost:5000`
   * **Z iného PC/mobilu v rovnakej sieti:** `http://<IP_ADRESA_JETSONU>:5000` (IP adresu zistíš príkazom `hostname -I`)

#### Funkcie webového rozhrania:
* **Overenie prístupu (Autentifikácia):** Umožňuje overiť osobu z nahranej fotografie, z webkamery v prehliadači alebo priamo snímkou z kamery pripojenej k Jetsonu. Posuvníkom **Tolerance** (`0.35 – 0.65`) je možné meniť prísnosť zhody.
* **Registrácia používateľa:** Pridá novú referenčnú snímku (zo súboru alebo kamery) priamo do `Attendance_data/<MENO>/` a okamžite aktualizuje cache vektorov. Jeden používateľ môže mať uložených viacero snímok z rôznych uhlov.
* **Synchronizácia s `capture.py`:** Tlačidlo **`↻ Načítať nové dáta z capture.py`** načíta a zaindexuje nové fotografie vytvorené cez terminálový skript.
* **História overení:** Tabuľka posledných pokusov o prístup s časovou pečiatkou, zdrojom snímky a nameranou vzdialenosťou.

### B. Rýchly zber referenčných snímok kamerou (`capture.py`)

Ak máš k Jetsonu pripojenú CSI alebo USB kameru a chceš rýchlo nasnímať sadu fotografií (predvolene 10 snímok na osobu):

```bash
python3 capture.py
```

**Ovládanie počas snímania:**
* `MEDZERNÍK (SPACE)` – zachytí snímku (vyžaduje presne 1 tvár v obraze s minimálnou veľkosťou 100×100 px)
* `r` – vymaže poslednú zachytenú snímku
* `q` – predčasne ukončí snímanie

**Voliteľné parametre CLI:**
```bash
python3 capture.py --max_images 5 --camera_id 0 --frame_width 640 --frame_height 480
```

### C. Dávkové testovanie na fotografiách (`test_photos.py`, `testphotoscas.py`)

Na otestovanie presnosti a výkonu bez kamery vlož referenčné fotky do `test_auth/authorized/` a testovacie fotky do `test_auth/test_inputs/`, potom spusti:

```bash
# Základné dávkové overenie s uložením výsledných snímok
python3 test_photos.py

# Dávkové overenie rozšírené o meranie času spracovania každej fotky a zápis logu
python3 testphotoscas.py
```

Oba skripty uložia výsledné snímky s vykreslenými rámčekmi a hodnotou `dist` do `test_auth/result_*.jpg`. Skript `testphotoscas.py` navyše meria čas spracovania každej snímky, vypočíta priemerný čas na 1 fotografiu a celý výpis zapíše do súboru `test_auth/log_vysledkov.txt`.

## 6. Dokumentácia kľúčových modulov

### `web_app.py`
* `sync_database_from_disk()` – prechádza podpriečinky v `Attendance_data/`, porovnáva čas úpravy a veľkosť súborov s `.encodings_cache.pkl` a počíta 128-D vektory iba pre nové alebo zmenené fotografie.
* `resize_if_needed(bgr_img, max_dim=1280)` – automaticky zmenšuje snímky s vysokým rozlíšením (napr. z mobilných telefónov), čím chráni 8 GB zdieľanej RAM na Jetsone pred preplnením pri CNN detekcii.
* `grab_frame_from_jetson_camera()` – inicializuje hardvérový GStreamer pipeline (`nvarguscamerasrc`) pre MIPI CSI kameru alebo automaticky prepne na dostupné `/dev/video*` USB zariadenie.
* `POST /register` – overí, že na vstupnej snímke sa nachádza práve jedna tvár, uloží snímku vo formáte `<MENO>_<ID>.jpg` a aktualizuje pamäť aj cache.
* `POST /verify` – deteguje všetky tváre v obraze cez GPU CNN model, porovná ich voči všetkým vektorom v databáze, vykreslí farebné ohraničenie (zelené = autorizovaný, červené = neautorizovaný) a zapíše záznam do `web_auth_log.csv`.

### `capture.py`
* `detect_faces_dnn(frame, net)` – vykonáva detekciu tvárí v reálnom čase na zmenšenej snímke (`RESIZE_SCALE = 0.4`) pomocou OpenCV DNN s CUDA akceleráciou.
* `initial_data_capture()` – riadi interaktívnu registráciu jednej osoby vrátane možnosti doplniť (`[A]dd`) alebo nahradiť (`[R]eplace`) existujúce snímky.
* `batch_capture()` – umožňuje hromadnú registráciu viacerých osôb za sebou bez reštartovania skriptu.

## 7. Riešenie častých problémov

* **Prehliadač blokuje webkameru pri prístupe cez IP adresu (`http://192.168.x.x:5000`):**
  Moderné prehliadače povoľujú prístup k lokálnej kamere `getUserMedia` iba cez `localhost` alebo `https://`. Pri pripojení zo vzdialeného PC použi na stránke tlačidlo **`🔌 Odfotiť z kamery v Jetsone`** (sníma kameru pripojenú priamo v Jetsone), alebo v prehliadači Chrome na klientskom PC pridaj adresu Jetsonu do výnimiek cez `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

* **`DLIB_USE_CUDA` vracia `False`:**
  Pravdepodobne došlo k prepísaniu ručne skompilovaného `dlib` príkazom `pip install -r requirements.txt`. Zopakuj Krok 3 z časti *Inštalácia a kompilácia s podporou CUDA*.

* **Chyba `ImportError` alebo `No module named 'pkg_resources'`:**
  Skontroluj, či máš v prostredí `dlib_env` nainštalované kompatibilné verzie: `pip install "setuptools<80" "numpy==1.26.4" "opencv-python<5"`.

## 8. Zdroje a knižničný základ

Knižničný základ (`face_recognition`, `dlib` s podporou CUDA, `opencv-python`) a postup konfigurácie prostredia pre platformu NVIDIA Jetson boli prevzaté z repozitára [**lovnishverma/FaceRecognition-Attendance**](https://github.com/lovnishverma/FaceRecognition-Attendance).
