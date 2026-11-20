import os
import cv2
import numpy as np
import face_recognition

AUTH_DIR = "test_auth/authorized"
TEST_DIR = "test_auth/test_inputs"
TOLERANCE = 0.50  # Nižšia hodnota = prísnejšia zhoda (odporúčané 0.45 - 0.55)

known_encodings = []
known_names = []

print("--- Načítavam autorizované osoby ---")
for fname in os.listdir(AUTH_DIR):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(AUTH_DIR, fname)
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(os.path.splitext(fname)[0])
            print(f"[OK] Registrovaný: {known_names[-1]}")
        else:
            print(f"[CHYBA] Nenašla sa tvár v: {fname}")

print("\n--- Testujem vstupné fotografie (CUDA CNN detektor) ---")
for fname in os.listdir(TEST_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(TEST_DIR, fname)
    rgb_img = face_recognition.load_image_file(path)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # model="cnn" využíva GPU (CUDA) na Jetson Orin Nano
    locations = face_recognition.face_locations(rgb_img, model="cnn")
    encodings = face_recognition.face_encodings(rgb_img, locations)

    print(f"Súbor {fname}: nájdených tvárí = {len(locations)}")

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

        print(f"  -> {label}")
        cv2.rectangle(bgr_img, (left, top), (right, bottom), color, 2)
        cv2.putText(bgr_img, label, (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = os.path.join("test_auth", f"result_{fname}")
    cv2.imwrite(out_path, bgr_img)