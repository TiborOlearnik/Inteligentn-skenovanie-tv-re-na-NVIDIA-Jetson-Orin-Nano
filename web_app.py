import os
import csv
import base64
import pickle
import shutil
import platform
from datetime import datetime
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Cesty zosúladené s capture.py (priečinok Attendance_data v koreni repozitára)
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_DIR, "Attendance_data")
CACHE_FILE = os.path.join(DATA_DIR, ".encodings_cache.pkl")
LOG_FILE = os.path.join(DATA_DIR, "web_auth_log.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# Štruktúra v pamäti:
# users_db = {
#     "PETER": {
#         "encodings": [np.array(...), ...],
#         "images": ["PETER_1.jpg", "PETER_2.jpg", ...]
#     }
# }
users_db = {}


def is_jetson():
    return platform.system() == "Linux" and "tegra" in platform.uname().release.lower()


def resize_if_needed(bgr_img, max_dim=1280):
    """Zmenší príliš veľké fotky, aby šetril 8 GB zdieľanej RAM na Jetsone."""
    h, w = bgr_img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        bgr_img = cv2.resize(bgr_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr_img


def load_cache():
    if os.path.isfile(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache_data):
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"[VAROVANIE] Nepodarilo sa uložiť cache: {e}")


def sync_database_from_disk():
    """
    Načíta všetkých používateľov a všetky ich referenčné fotky z Attendance_data/<MENO>/.
    Využíva cache, aby sa už raz spracované fotky z capture.py nemuseli znova počítať.
    """
    users_db.clear()
    cache = load_cache()
    new_cache = {}
    encoded_new_count = 0

    if not os.path.exists(DATA_DIR):
        return

    for person_name in sorted(os.listdir(DATA_DIR)):
        person_dir = os.path.join(DATA_DIR, person_name)
        if not os.path.isdir(person_dir) or person_name.startswith("."):
            continue

        person_encodings = []
        person_images = []

        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full_path = os.path.join(person_dir, fname)
            rel_key = f"{person_name}/{fname}"
            stat = os.stat(full_path)
            sig = (stat.st_mtime, stat.st_size)

            if rel_key in cache and cache[rel_key]["sig"] == sig:
                encoding = cache[rel_key]["encoding"]
            else:
                bgr = cv2.imread(full_path)
                if bgr is None:
                    continue
                bgr = resize_if_needed(bgr)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb, model="cnn")
                if not locs:
                    # Fallback na HOG, ak je fotka z capture.py menšia
                    locs = face_recognition.face_locations(rgb, model="hog")
                encs = face_recognition.face_encodings(rgb, locs)
                if not encs:
                    print(f"[PRESKOČENÉ] Nenašla sa tvár v {rel_key}")
                    continue
                encoding = encs[0]
                encoded_new_count += 1

            new_cache[rel_key] = {"sig": sig, "encoding": encoding}
            person_encodings.append(encoding)
            person_images.append(fname)

        if person_encodings:
            users_db[person_name] = {
                "encodings": person_encodings,
                "images": person_images
            }
            print(f"[OK] Načítaný používateľ: {person_name} ({len(person_images)} fotiek)")

    save_cache(new_cache)
    if encoded_new_count > 0:
        print(f"[INFO] Vypočítaných {encoded_new_count} nových vektorov tvárí.")


def decode_uploaded_or_camera_image(req):
    """Načíta obrázok buď z nahratého súboru alebo z base64 snímky z kamery."""
    camera_b64 = req.form.get("camera_image", "").strip()
    if camera_b64:
        if "," in camera_b64:
            camera_b64 = camera_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(camera_b64)
            return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None

    file = req.files.get("photo")
    if file and file.filename:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return None


def grab_frame_from_jetson_camera(camera_id=0, width=640, height=480):
    """Zachytí jednu snímku z kamery pripojenej priamo k Jetsonu (CSI GStreamer alebo USB)."""
    cap = None
    if is_jetson():
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={camera_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            cap = None

    if cap is None or not cap.isOpened():
        for idx in range(3):
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ret, frame = test_cap.read()
                test_cap.release()
                if ret and frame is not None:
                    return frame
        return None

    # Zahodíme prvé 2 snímky kvôli automatickej expozícii
    frame = None
    for _ in range(3):
        ret, f = cap.read()
        if ret:
            frame = f
    cap.release()
    return frame


def log_attempt(status, name, distance, source):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Cas", "Zdroj", "Stav", "Osoba", "Vzdialenost"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source,
            status,
            name,
            f"{distance:.3f}" if distance is not None else "-"
        ])


def read_logs(limit=12):
    if not os.path.isfile(LOG_FILE):
        return []
    with open(LOG_FILE, mode="r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return []
    return list(reversed(rows[1:]))[:limit]


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="sk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jetson Orin Nano - Face Auth &amp; Attendance</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
        .container { max-width: 1180px; margin: 0 auto; }
        .header-row { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
        h1 { color: #38bdf8; margin: 0 0 4px 0; font-size: 1.6rem; }
        .subtitle { color: #94a3b8; font-size: 0.92rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        h2 { margin-top: 0; font-size: 1.15rem; color: #f8fafc; border-bottom: 1px solid #334155; padding-bottom: 10px; }
        label { display: block; margin: 12px 0 6px; font-weight: 600; font-size: 0.9rem; }
        input[type="text"], input[type="file"], input[type="range"] {
            width: 100%; box-sizing: border-box; padding: 10px; border-radius: 8px;
            border: 1px solid #475569; background: #0f172a; color: #f8fafc;
        }
        .tabs { display: flex; gap: 8px; margin-bottom: 12px; }
        .tab-btn {
            flex: 1; padding: 9px; border: 1px solid #475569; border-radius: 8px;
            background: #0f172a; color: #cbd5e1; font-weight: 600; cursor: pointer; margin-top: 0;
        }
        .tab-btn.active { background: #0284c7; border-color: #38bdf8; color: white; }
        .btn-primary {
            margin-top: 15px; width: 100%; padding: 12px; border: none; border-radius: 8px;
            background: #0284c7; color: white; font-weight: 700; font-size: 1rem; cursor: pointer;
        }
        .btn-primary:hover { background: #0369a1; }
        .btn-green { background: #059669; }
        .btn-green:hover { background: #047857; }
        .btn-secondary {
            padding: 8px 14px; border: 1px solid #475569; border-radius: 8px;
            background: #334155; color: #f8fafc; font-weight: 600; cursor: pointer; font-size: 0.88rem;
        }
        .btn-secondary:hover { background: #475569; }
        .btn-danger {
            background: #dc2626; border: none; border-radius: 6px; color: white;
            padding: 6px 12px; font-size: 0.82rem; font-weight: 600; cursor: pointer;
        }
        .btn-danger:hover { background: #b91c1c; }
        .cam-box {
            background: #0f172a; border: 1px dashed #475569; border-radius: 8px;
            padding: 12px; text-align: center; margin-top: 8px;
        }
        video, .snapshot-preview { width: 100%; max-height: 280px; border-radius: 8px; background: #000; object-fit: cover; }
        .cam-controls { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
        .cam-controls button { flex: 1; margin-top: 0; }
        .alert { padding: 12px 15px; border-radius: 8px; margin-bottom: 20px; font-weight: 600; }
        .alert-ok { background: #064e3b; border: 1px solid #10b981; color: #a7f3d0; }
        .alert-err { background: #7f1d1d; border: 1px solid #ef4444; color: #fecaca; }
        .result-box { margin-top: 18px; text-align: center; }
        .result-box img { max-width: 100%; border-radius: 8px; border: 2px solid #475569; }
        .user-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(145px, 1fr)); gap: 12px; }
        .user-item { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; text-align: center; }
        .user-item img { width: 82px; height: 82px; object-fit: cover; border-radius: 50%; border: 2px solid #38bdf8; }
        .user-name { font-weight: 700; margin: 6px 0 2px; font-size: 0.95rem; }
        .user-count { color: #94a3b8; font-size: 0.8rem; margin-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
        th, td { text-align: left; padding: 9px; border-bottom: 1px solid #334155; }
        th { color: #94a3b8; }
        .badge-ok { color: #34d399; font-weight: 700; }
        .badge-no { color: #f87171; font-weight: 700; }
        .hidden { display: none; }
    </style>
</head>
<body>
<div class="container">
    <div class="header-row">
        <div>
            <h1>Jetson Orin Nano — Autentifikácia tváre</h1>
            <div class="subtitle">
                Databáza: <code>Attendance_data/</code> (zdieľaná s <code>capture.py</code>) •
                Osôb: <strong>{{ users|length }}</strong> •
                Referenčných fotiek: <strong>{{ total_images }}</strong>
            </div>
        </div>
        <form action="/reload" method="post">
            <button type="submit" class="btn-secondary">↻ Načítať nové dáta z capture.py</button>
        </form>
    </div>

    {% if message %}
        <div class="alert {{ 'alert-ok' if success else 'alert-err' }}">{{ message }}</div>
    {% endif %}

    <div class="grid">
        <!-- 1. OVERENIE TVÁRE -->
        <div class="card">
            <h2>1. Overenie prístupu (Autentifikácia)</h2>
            <div class="tabs">
                <button type="button" id="tab-ver-file" class="tab-btn active" onclick="switchMode('ver', 'file')">📁 Nahrať fotku</button>
                <button type="button" id="tab-ver-cam" class="tab-btn" onclick="switchMode('ver', 'cam')">📷 Použiť kameru</button>
            </div>

            <form action="/verify" method="post" enctype="multipart/form-data" onsubmit="return validateForm('ver')">
                <input type="hidden" id="ver_camera_image" name="camera_image" value="">

                <div id="ver-file-box">
                    <label for="verify_photo">Vyber fotografiu zo súboru:</label>
                    <input type="file" id="verify_photo" name="photo" accept="image/*">
                </div>

                <div id="ver-cam-box" class="cam-box hidden">
                    <video id="ver_video" autoplay playsinline class="hidden"></video>
                    <img id="ver_preview" class="snapshot-preview hidden" alt="Náhľad snímky">
                    <canvas id="ver_canvas" class="hidden"></canvas>
                    <div id="ver_cam_status" style="color:#94a3b8; font-size:0.85rem; margin:6px 0;">
                        Vyber zdroj kamery nižšie:
                    </div>
                    <div class="cam-controls">
                        <button type="button" class="btn-secondary" onclick="startBrowserCam('ver')">▶ Zapnúť webkameru</button>
                        <button type="button" class="btn-secondary" onclick="snapBrowserCam('ver')">📸 Odfotiť z webkamery</button>
                        <button type="button" class="btn-secondary" onclick="snapJetsonCam('ver')">🔌 Odfotiť z kamery v Jetsone</button>
                    </div>
                </div>

                <label for="tolerance">Prísnosť porovnania (Tolerance: <span id="tol_val">{{ tolerance }}</span>):</label>
                <input type="range" id="tolerance" name="tolerance" min="0.35" max="0.65" step="0.01"
                       value="{{ tolerance }}" oninput="document.getElementById('tol_val').innerText = this.value">
                <small style="color:#94a3b8;">Nižšia hodnota = prísnejšia zhoda (odporúčané 0.50)</small>

                <button type="submit" class="btn-primary">Overiť tvár cez GPU</button>
            </form>

            {% if result_image %}
            <div class="result-box">
                <img src="data:image/jpeg;base64,{{ result_image }}" alt="Výsledok overenia">
            </div>
            {% endif %}
        </div>

        <!-- 2. REGISTRÁCIA POUŽÍVATEĽA -->
        <div class="card">
            <h2>2. Registrácia používateľa (do Attendance_data)</h2>
            <div class="tabs">
                <button type="button" id="tab-reg-file" class="tab-btn active" onclick="switchMode('reg', 'file')">📁 Nahrať fotku</button>
                <button type="button" id="tab-reg-cam" class="tab-btn" onclick="switchMode('reg', 'cam')">📷 Použiť kameru</button>
            </div>

            <form action="/register" method="post" enctype="multipart/form-data" onsubmit="return validateForm('reg')">
                <label for="name">Meno používateľa (uloží sa veľkými písmenami ako v capture.py):</label>
                <input type="text" id="name" name="name" placeholder="napr. PETER" required>

                <input type="hidden" id="reg_camera_image" name="camera_image" value="">

                <div id="reg-file-box">
                    <label for="reg_photo">Referenčná fotografia (1 tvár spredu):</label>
                    <input type="file" id="reg_photo" name="photo" accept="image/*">
                </div>

                <div id="reg-cam-box" class="cam-box hidden">
                    <video id="reg_video" autoplay playsinline class="hidden"></video>
                    <img id="reg_preview" class="snapshot-preview hidden" alt="Náhľad snímky">
                    <canvas id="reg_canvas" class="hidden"></canvas>
                    <div id="reg_cam_status" style="color:#94a3b8; font-size:0.85rem; margin:6px 0;">
                        Vyber zdroj kamery nižšie:
                    </div>
                    <div class="cam-controls">
                        <button type="button" class="btn-secondary" onclick="startBrowserCam('reg')">▶ Zapnúť webkameru</button>
                        <button type="button" class="btn-secondary" onclick="snapBrowserCam('reg')">📸 Odfotiť z webkamery</button>
                        <button type="button" class="btn-secondary" onclick="snapJetsonCam('reg')">🔌 Odfotiť z kamery v Jetsone</button>
                    </div>
                </div>

                <button type="submit" class="btn-primary btn-green">Uložiť do databázy</button>
            </form>

            <h2 style="margin-top: 28px;">Registrovaní používatelia</h2>
            {% if users %}
            <div class="user-list">
                {% for name, info in users.items() %}
                <div class="user-item">
                    <img src="{{ url_for('person_photo', person=name, filename=info.images[0]) }}" alt="{{ name }}">
                    <div class="user-name">{{ name }}</div>
                    <div class="user-count">{{ info.images|length }} {{ 'fotka' if info.images|length == 1 else ('fotky' if info.images|length < 5 else 'fotiek') }}</div>
                    <form action="/delete/{{ name }}" method="post" onsubmit="return confirm('Naozaj vymazať osobu {{ name }} a všetky jej fotky?');">
                        <button type="submit" class="btn-danger">Odstrániť</button>
                    </form>
                </div>
                {% endfor %}
            </div>
            {% else %}
                <p style="color:#94a3b8;">Priečinok <code>Attendance_data/</code> je zatiaľ prázdny.</p>
            {% endif %}
        </div>
    </div>

    <!-- HISTÓRIA OVERENÍ -->
    <div class="card">
        <h2>Posledné pokusy o overenie</h2>
        {% if logs %}
        <table>
            <thead>
                <tr><th>Čas</th><th>Zdroj</th><th>Výsledok</th><th>Identifikovaná osoba</th><th>Najmenšia vzdialenosť</th></tr>
            </thead>
            <tbody>
                {% for row in logs %}
                <tr>
                    <td>{{ row[0] }}</td>
                    <td>{{ row[1] }}</td>
                    <td class="{{ 'badge-ok' if row[2] == 'AUTORIZOVANY' else 'badge-no' }}">{{ row[2] }}</td>
                    <td>{{ row[3] }}</td>
                    <td>{{ row[4] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
            <p style="color:#94a3b8;">Zatiaľ neprebehlo žiadne overenie.</p>
        {% endif %}
    </div>
</div>

<script>
const activeStreams = { ver: null, reg: null };

function switchMode(prefix, mode) {
    const fileBox = document.getElementById(prefix + '-file-box');
    const camBox = document.getElementById(prefix + '-cam-box');
    const tabFile = document.getElementById('tab-' + prefix + '-file');
    const tabCam = document.getElementById('tab-' + prefix + '-cam');

    if (mode === 'file') {
        fileBox.classList.remove('hidden');
        camBox.classList.add('hidden');
        tabFile.classList.add('active');
        tabCam.classList.remove('active');
        document.getElementById(prefix + '_camera_image').value = '';
        stopBrowserCam(prefix);
    } else {
        fileBox.classList.add('hidden');
        camBox.classList.remove('hidden');
        tabFile.classList.remove('active');
        tabCam.classList.add('active');
        const fileInput = document.getElementById(prefix === 'ver' ? 'verify_photo' : 'reg_photo');
        fileInput.value = '';
    }
}

function stopBrowserCam(prefix) {
    if (activeStreams[prefix]) {
        activeStreams[prefix].getTracks().forEach(t => t.stop());
        activeStreams[prefix] = null;
    }
}

async function startBrowserCam(prefix) {
    const statusEl = document.getElementById(prefix + '_cam_status');
    const videoEl = document.getElementById(prefix + '_video');
    const previewEl = document.getElementById(prefix + '_preview');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        statusEl.innerText = "Prehliadač blokuje webkameru (cez sieť HTTP použi tlačidlo 'Odfotiť z kamery v Jetsone' alebo otvor stránku cez localhost).";
        return;
    }

    try {
        stopBrowserCam(prefix);
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
        activeStreams[prefix] = stream;
        videoEl.srcObject = stream;
        videoEl.classList.remove('hidden');
        previewEl.classList.add('hidden');
        statusEl.innerText = "Webkamera je aktívna. Klikni na 'Odfotiť z webkamery'.";
    } catch (err) {
        statusEl.innerText = "Nepodarilo sa spustiť webkameru: " + err.message;
    }
}

function snapBrowserCam(prefix) {
    const videoEl = document.getElementById(prefix + '_video');
    const canvasEl = document.getElementById(prefix + '_canvas');
    const previewEl = document.getElementById(prefix + '_preview');
    const hiddenInput = document.getElementById(prefix + '_camera_image');
    const statusEl = document.getElementById(prefix + '_cam_status');

    if (!activeStreams[prefix] || videoEl.videoWidth === 0) {
        statusEl.innerText = "Najprv klikni na 'Zapnúť webkameru'.";
        return;
    }

    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    const ctx = canvasEl.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

    const dataUrl = canvasEl.toDataURL('image/jpeg', 0.92);
    hiddenInput.value = dataUrl;
    previewEl.src = dataUrl;
    previewEl.classList.remove('hidden');
    videoEl.classList.add('hidden');
    stopBrowserCam(prefix);
    statusEl.innerText = "✓ Snímka z webkamery je pripravená na odoslanie.";
}

async function snapJetsonCam(prefix) {
    const statusEl = document.getElementById(prefix + '_cam_status');
    const videoEl = document.getElementById(prefix + '_video');
    const previewEl = document.getElementById(prefix + '_preview');
    const hiddenInput = document.getElementById(prefix + '_camera_image');

    statusEl.innerText = "Snímam obraz z kamery na Jetsone...";
    stopBrowserCam(prefix);
    videoEl.classList.add('hidden');

    try {
        const resp = await fetch('/jetson_snapshot', { method: 'POST' });
        const data = await resp.json();
        if (!data.ok) {
            statusEl.innerText = "Chyba kamery na Jetsone: " + data.error;
            return;
        }
        const dataUrl = "data:image/jpeg;base64," + data.image;
        hiddenInput.value = dataUrl;
        previewEl.src = dataUrl;
        previewEl.classList.remove('hidden');
        statusEl.innerText = "✓ Snímka z kamery na Jetsone je pripravená.";
    } catch (err) {
        statusEl.innerText = "Chyba komunikácie so serverom: " + err.message;
    }
}

function validateForm(prefix) {
    const camVal = document.getElementById(prefix + '_camera_image').value;
    const fileInput = document.getElementById(prefix === 'ver' ? 'verify_photo' : 'reg_photo');
    if (!camVal && (!fileInput.files || fileInput.files.length === 0)) {
        alert("Najprv vyber fotografiu zo súboru alebo urob snímku kamerou.");
        return false;
    }
    return true;
}
</script>
</body>
</html>
"""


def render_page(tolerance=0.50, message=None, success=True, result_image=None):
    total_images = sum(len(u["images"]) for u in users_db.values())
    return render_template_string(
        HTML_TEMPLATE,
        users=users_db,
        total_images=total_images,
        logs=read_logs(),
        tolerance=tolerance,
        message=message,
        success=success,
        result_image=result_image
    )


@app.route("/")
def index():
    return render_page()


@app.route("/person_photo/<person>/<filename>")
def person_photo(person, filename):
    safe_person = secure_filename(person).upper()
    safe_file = secure_filename(filename)
    return send_from_directory(os.path.join(DATA_DIR, safe_person), safe_file)


@app.route("/reload", methods=["POST"])
def reload_db():
    sync_database_from_disk()
    return render_page(message="Databáza bola obnovená z priečinka Attendance_data/.", success=True)


@app.route("/jetson_snapshot", methods=["POST"])
def jetson_snapshot():
    frame = grab_frame_from_jetson_camera()
    if frame is None:
        return jsonify({"ok": False, "error": "Na Jetsone sa nenašla žiadna pripojená CSI ani USB kamera."})
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return jsonify({"ok": True, "image": b64_str})


@app.route("/register", methods=["POST"])
def register():
    raw_name = request.form.get("name", "").strip()
    safe_name = secure_filename(raw_name).upper()
    if not safe_name:
        return render_page(message="Zadaj platné meno používateľa.", success=False)

    bgr_img = decode_uploaded_or_camera_image(request)
    if bgr_img is None:
        return render_page(message="Nepodarilo sa načítať obrázok zo súboru ani z kamery.", success=False)

    bgr_img = resize_if_needed(bgr_img)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb_img, model="cnn")
    if len(locations) != 1:
        msg = (
            "Na snímke sa nenašla žiadna tvár."
            if len(locations) == 0
            else f"Na snímke sa našlo viac tvárí ({len(locations)}). Zaregistruj naraz iba jednu osobu."
        )
        return render_page(message=msg, success=False)

    encodings = face_recognition.face_encodings(rgb_img, locations)

    # Uloženie do Attendance_data/<MENO>/<MENO>_<číslo>.jpg presne ako capture.py
    person_dir = os.path.join(DATA_DIR, safe_name)
    os.makedirs(person_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    next_idx = len(existing_files) + 1
    filename = f"{safe_name}_{next_idx}.jpg"
    while os.path.exists(os.path.join(person_dir, filename)):
        next_idx += 1
        filename = f"{safe_name}_{next_idx}.jpg"

    save_path = os.path.join(person_dir, filename)
    cv2.imwrite(save_path, bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # Aktualizácia pamäte a cache
    if safe_name not in users_db:
        users_db[safe_name] = {"encodings": [], "images": []}
    users_db[safe_name]["encodings"].append(encodings[0])
    users_db[safe_name]["images"].append(filename)

    cache = load_cache()
    stat = os.stat(save_path)
    cache[f"{safe_name}/{filename}"] = {
        "sig": (stat.st_mtime, stat.st_size),
        "encoding": encodings[0]
    }
    save_cache(cache)

    return render_page(
        message=f"Fotka '{filename}' pre osobu '{safe_name}' bola uložená do Attendance_data/.",
        success=True
    )


@app.route("/verify", methods=["POST"])
def verify():
    tolerance = float(request.form.get("tolerance", 0.50))
    source_type = "Kamera" if request.form.get("camera_image", "").strip() else "Súbor"

    if not users_db:
        return render_page(
            tolerance=tolerance,
            message="Databáza Attendance_data/ je prázdna. Najprv zaregistruj osobu alebo spusti capture.py.",
            success=False
        )

    bgr_img = decode_uploaded_or_camera_image(request)
    if bgr_img is None:
        return render_page(tolerance=tolerance, message="Nepodarilo sa načítať obrázok.", success=False)

    bgr_img = resize_if_needed(bgr_img)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb_img, model="cnn")
    encodings = face_recognition.face_encodings(rgb_img, locations)

    if not locations:
        log_attempt("BEZ_TVARE", "-", None, source_type)
        return render_page(tolerance=tolerance, message="Na snímke sa nenašla žiadna tvár.", success=False)

    # Zozbieranie všetkých referenčných vektorov zo všetkých fotiek všetkých osôb
    all_names = []
    all_encodings = []
    for person_name, info in users_db.items():
        for enc in info["encodings"]:
            all_names.append(person_name)
            all_encodings.append(enc)

    any_authorized = False
    summary_parts = []

    for (top, right, bottom, left), enc in zip(locations, encodings):
        distances = face_recognition.face_distance(all_encodings, enc)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])

        if best_dist <= tolerance:
            matched_name = all_names[best_idx]
            label = f"AUTORIZOVANY: {matched_name} ({best_dist:.2f})"
            color = (0, 255, 0)
            any_authorized = True
            summary_parts.append(f"{matched_name} (dist: {best_dist:.2f})")
            log_attempt("AUTORIZOVANY", matched_name, best_dist, source_type)
        else:
            label = f"NEAUTORIZOVANY ({best_dist:.2f})"
            color = (0, 0, 255)
            summary_parts.append(f"Neznáma osoba (dist: {best_dist:.2f})")
            log_attempt("NEAUTORIZOVANY", "Neznámy", best_dist, source_type)

        cv2.rectangle(bgr_img, (left, top), (right, bottom), color, 2)
        cv2.putText(bgr_img, label, (left, max(25, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    _, buffer = cv2.imencode(".jpg", bgr_img)
    result_b64 = base64.b64encode(buffer).decode("utf-8")

    return render_page(
        tolerance=tolerance,
        message="Výsledok overenia: " + ", ".join(summary_parts),
        success=any_authorized,
        result_image=result_b64
    )


@app.route("/delete/<name>", methods=["POST"])
def delete_user(name):
    safe_name = secure_filename(name).upper()
    person_dir = os.path.join(DATA_DIR, safe_name)
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
    if safe_name in users_db:
        del users_db[safe_name]
    sync_database_from_disk()
    return redirect(url_for("index"))


if __name__ == "__main__":
    sync_database_from_disk()
    app.run(host="0.0.0.0", port=5000, debug=False)