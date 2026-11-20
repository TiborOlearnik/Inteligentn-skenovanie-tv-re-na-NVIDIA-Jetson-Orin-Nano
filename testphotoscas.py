import os
import time
from datetime import datetime
import cv2
import numpy as np
import face_recognition

AUTH_DIR = "test_auth/authorized"
TEST_DIR = "test_auth/test_inputs"
LOG_FILE = "test_auth/log_vysledkov.txt"
TOLERANCE = 0.50  # Nižšia hodnota = prísnejšia zhoda (odporúčané 0.45 - 0.55)

known_encodings = []
known_names = []
log_lines = []


def log(msg=""):
    """Vypíše správu do konzoly a zároveň ju uloží do zoznamu pre zápis do súboru."""
    print(msg)
    log_lines.append(msg)


os.makedirs("test_auth", exist_ok=True)
log(f"=== Spustenie testu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

log("--- Načítavam autorizované osoby ---")
for fname in os.listdir(AUTH_DIR):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(AUTH_DIR, fname)
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(os.path.splitext(fname)[0])
            log(f"[OK] Registrovaný: {known_names[-1]}")
        else:
            log(f"[CHYBA] Nenašla sa tvár v: {fname}")

log("\n--- Testujem vstupné fotografie (CUDA CNN detektor) ---")

processed_count = 0
start_time = time.perf_counter()

for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_start = time.perf_counter()
    path = os.path.join(TEST_DIR, fname)
    rgb_img = face_recognition.load_image_file(path)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # model="cnn" využíva GPU (CUDA) na Jetson Orin Nano
    locations = face_recognition.face_locations(rgb_img, model="cnn")
    encodings = face_recognition.face_encodings(rgb_img, locations)

    img_elapsed = time.perf_counter() - img_start
    processed_count += 1

    log(f"Súbor {fname}: nájdených tvárí = {len(locations)} (čas: {img_elapsed:.3f} s)")

    for (top, right, bottom, left), enc in zip(locations, encodings):
        distances = face_recognition.face_distance(known_encodings, enc)
        best_idx = int(np.argmin(distances))
        best_dist = distances[best_idx]

        if best_dist <= TOLERANCE:
            label = f"AUTORIZOVANY: {known_names[best_idx]} (dist: {best_dist:.2f})"
            color = (0, 255, 0)
        else:
            label = f"NEAUTORIZOVANY (dist: {best_dist:.2f})"
            color = (0, 0, 255)

        log(f"  -> {label}")
        cv2.rectangle(bgr_img, (left, top), (right, bottom), color, 2)
        cv2.putText(bgr_img, label, (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = os.path.join("test_auth", f"result_{fname}")
    cv2.imwrite(out_path, bgr_img)

total_elapsed = time.perf_counter() - start_time
avg_time = (total_elapsed / processed_count) if processed_count > 0 else 0.0

log("\n--- Súhrn ---")
log(f"Spracovaných fotografií: {processed_count}")
log(f"Celkový čas spracovania: {total_elapsed:.3f} s")
log(f"Priemerný čas na 1 fotografiu: {avg_time:.3f} s")
log("=" * 45 + "\n")

# Uloženie výpisu do textového súboru (režim "a" pripíše nové behy na koniec súboru;
# ak chceš súbor pri každom spustení prepísať, zmeň "a" na "w")
with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")

print(f"Výpis bol uložený do: {LOG_FILE}")
