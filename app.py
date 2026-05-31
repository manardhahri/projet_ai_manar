# -*- coding: utf-8 -*-
"""
DermoAI - Flask application for skin lesion classification.

The app follows the requested TP routes:
  /          authentication
  /dashboard protected dashboard
  /predict   image upload + VGG16 prediction
  /patients  diagnosis history
  /logout    session reset
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

try:
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.preprocessing import image

    TENSORFLOW_AVAILABLE = True
except Exception as import_error:  # pragma: no cover - useful for classroom machines.
    tf = None
    VGG16 = Dense = Dropout = Flatten = Input = Model = load_model = image = None
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_IMPORT_ERROR = import_error

try:
    from PIL import Image, ImageFilter, ImageStat
except Exception as import_error:  # pragma: no cover
    Image = ImageFilter = ImageStat = None
    PIL_IMPORT_ERROR = import_error


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
DB_PATH = BASE_DIR / "skin_cancer.db"
MODEL_CANDIDATES = [
    BASE_DIR / "model" / "vgg16_skin_cancer.h5",
    BASE_DIR / "model" / "mon_modele_melanome_vgg16.h5",
]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.secret_key = os.environ.get("DERMOAI_SECRET_KEY", "dermoai_secret_2026")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def get_db() -> sqlite3.Connection:
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_db(_exception: Exception | None) -> None:
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = sqlite3.connect(DB_PATH)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            result TEXT NOT NULL,
            probability REAL NOT NULL,
            image_path TEXT NOT NULL,
            gradcam_path TEXT,
            heatmap_path TEXT,
            attention_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    ensure_column(db, "patients", "gradcam_path", "TEXT")
    ensure_column(db, "patients", "heatmap_path", "TEXT")
    ensure_column(db, "patients", "attention_path", "TEXT")
    db.execute(
        "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
        ("admin", "1234"),
    )
    db.commit()
    db.close()


def ensure_column(db: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def model_path() -> Path | None:
    return next((path for path in MODEL_CANDIDATES if path.exists()), None)


def build_vgg16_model(weights_path: Path):
    inputs = Input(shape=(224, 224, 3), name="input_layer")
    base = VGG16(weights=None, include_top=False, input_tensor=inputs)
    x = Flatten(name="flatten")(base.output)
    x = Dense(256, activation="relu", name="dense")(x)
    x = Dropout(0.5, name="dropout")(x)
    outputs = Dense(1, activation="sigmoid", name="dense_1")(x)
    mdl = Model(inputs=inputs, outputs=outputs)
    mdl.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
    return mdl


def load_ai_model():
    path = model_path()
    if not TENSORFLOW_AVAILABLE or path is None:
        return None, "demo"

    try:
        return load_model(str(path), compile=False), "tensorflow"
    except Exception:
        return build_vgg16_model(path), "tensorflow"


init_db()
model, MODEL_MODE = load_ai_model()


def login_required():
    return "user" in session


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_upload_name(filename: str) -> str:
    safe_name = secure_filename(filename)
    stem = Path(safe_name).stem or "lesion"
    suffix = Path(safe_name).suffix.lower() or ".jpg"
    return f"{stem}_{uuid.uuid4().hex[:10]}{suffix}"


def stored_to_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    path_obj = Path(path)
    try:
        rel = path_obj.resolve().relative_to(BASE_DIR)
    except Exception:
        rel = path_obj
    return rel.as_posix()


def prepare_image(path: Path) -> np.ndarray:
    if TENSORFLOW_AVAILABLE:
        img = image.load_img(str(path), target_size=(224, 224))
        img_arr = image.img_to_array(img)
    else:
        if Image is None:
            raise RuntimeError(f"Pillow is required for image processing: {PIL_IMPORT_ERROR}")
        img_arr = np.array(Image.open(path).convert("RGB").resize((224, 224)))
    return np.expand_dims(img_arr / 255.0, axis=0)


def demo_prediction(path: Path) -> float:
    digest = hashlib.sha256(path.read_bytes()).digest()
    seed = int.from_bytes(digest[:4], "big")
    return 0.12 + (seed % 7600) / 10000


def assess_image_quality(path: Path) -> dict[str, object]:
    if Image is None or ImageFilter is None or ImageStat is None:
        return {
            "score": 0,
            "label": "Unavailable",
            "tips": ["Pillow is required for image quality analysis."],
        }

    img = Image.open(path).convert("RGB")
    width, height = img.size
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]
    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = ImageStat.Stat(edges).stddev[0]

    score = 100
    tips: list[str] = []
    if min(width, height) < 224:
        score -= 24
        tips.append("Use an image of at least 224 x 224 pixels.")
    if brightness < 55:
        score -= 18
        tips.append("Image is dark; improve lighting before clinical review.")
    elif brightness > 205:
        score -= 18
        tips.append("Image is overexposed; reduce glare and flash.")
    if contrast < 28:
        score -= 18
        tips.append("Low contrast detected; center the lesion on clear skin.")
    if sharpness < 16:
        score -= 22
        tips.append("Possible blur detected; retake with steadier focus.")
    if not tips:
        tips.append("Image quality looks suitable for AI screening.")

    score = max(0, min(100, int(round(score))))
    label = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 55 else "Needs review"
    return {
        "score": score,
        "label": label,
        "tips": tips,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness": round(sharpness, 1),
        "resolution": f"{width} x {height}",
    }


def build_result_analysis(result: str, probability: float, image_quality: dict[str, object]) -> dict[str, object]:
    confidence = probability * 100
    quality_score = int(image_quality.get("score", 0)) if image_quality else 0

    if result == "Malignant":
        title = "Suspicious lesion pattern detected"
        summary = (
            "The model found visual patterns that are closer to malignant examples in its training data. "
            "This does not replace a clinical diagnosis, but it should be treated as a priority case for review."
        )
        actions = [
            "Refer the patient to a dermatologist or qualified clinician for confirmation.",
            "Compare the lesion with previous images if available, especially changes in size, border, or color.",
            "Keep the original image and AI report in the patient record.",
        ]
    else:
        title = "Benign-leaning lesion pattern"
        summary = (
            "The model found visual patterns that are closer to benign examples. "
            "This result is reassuring, but continued observation is recommended if the lesion changes."
        )
        actions = [
            "Recommend routine monitoring and patient education about visible changes.",
            "Repeat the scan with a clearer image if the lesion is new, evolving, painful, or bleeding.",
            "Escalate to medical review when clinical symptoms conflict with the AI result.",
        ]

    if confidence >= 80:
        confidence_note = "The model confidence is high, so the prediction is internally consistent."
    elif confidence >= 60:
        confidence_note = "The model confidence is moderate; review the image and clinical context carefully."
    else:
        confidence_note = "The model confidence is low; this result should be considered uncertain."

    if quality_score >= 75:
        quality_note = "Image quality is acceptable for screening."
    elif quality_score >= 55:
        quality_note = "Image quality is usable, but a sharper and better-lit photo would improve reliability."
    else:
        quality_note = "Image quality is weak; retake the photo before relying on the analysis."

    return {
        "title": title,
        "summary": summary,
        "confidence_note": confidence_note,
        "quality_note": quality_note,
        "actions": actions,
    }


def predict_probability(path: Path, img_arr: np.ndarray) -> float:
    if model is None:
        return float(demo_prediction(path))
    return float(model.predict(img_arr, verbose=0)[0][0])


def generate_gradcam_heatmap(mdl, img_array: np.ndarray, layer_name: str = "block5_conv3"):
    if not TENSORFLOW_AVAILABLE or mdl is None:
        return None
    grad_model = tf.keras.models.Model(
        inputs=mdl.inputs,
        outputs=[mdl.get_layer(layer_name).output, mdl.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_score = preds[:, 0]
    grads = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    h_img = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
    return np.array(h_img.resize(size, Image.BILINEAR)).astype(np.float32) / 255.0


def apply_colormap(heatmap_2d: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(cmap_name)
        return (cmap(heatmap_2d)[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        red = np.clip(1.5 - np.abs(4 * heatmap_2d - 3), 0, 1)
        green = np.clip(1.5 - np.abs(4 * heatmap_2d - 2), 0, 1)
        blue = np.clip(1.5 - np.abs(4 * heatmap_2d - 1), 0, 1)
        return (np.dstack([red, green, blue]) * 255).astype(np.uint8)


def save_xai_images(original_path: Path, filename_stem: str, img_arr: np.ndarray) -> dict[str, str | None]:
    paths = {"gradcam": None, "heatmap": None, "attention": None}
    if Image is None or model is None:
        return paths

    try:
        heatmap = generate_gradcam_heatmap(model, img_arr)
        if heatmap is None:
            return paths

        original = Image.open(original_path).convert("RGB").resize((224, 224))
        original_arr = np.array(original, dtype=np.float32)
        resized = resize_heatmap(heatmap)
        colored = apply_colormap(resized, "jet").astype(np.float32)

        gradcam = np.clip(original_arr * 0.55 + colored * 0.45, 0, 255).astype(np.uint8)
        attention = np.clip(original_arr * (0.15 + 0.85 * resized[:, :, np.newaxis]), 0, 255).astype(np.uint8)
        pure = apply_colormap(resized, "inferno")

        outputs = {
            "gradcam": UPLOAD_FOLDER / f"gradcam_{filename_stem}.png",
            "heatmap": UPLOAD_FOLDER / f"heatmap_{filename_stem}.png",
            "attention": UPLOAD_FOLDER / f"attention_{filename_stem}.png",
        }
        Image.fromarray(gradcam).save(outputs["gradcam"])
        Image.fromarray(pure).save(outputs["heatmap"])
        Image.fromarray(attention).save(outputs["attention"])
        return {key: stored_to_url(value) for key, value in outputs.items()}
    except Exception as xai_error:
        print(f"[XAI] Grad-CAM unavailable: {xai_error}")
        return paths


@app.route("/", methods=["GET", "POST"])
def login():
    if login_required():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        row = get_db().execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        if row:
            session["user"] = username
            flash("Connexion reussie.", "success")
            return redirect(url_for("dashboard"))
        flash("Identifiant ou mot de passe incorrect.", "danger")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    malignant = db.execute("SELECT COUNT(*) FROM patients WHERE result = 'Malignant'").fetchone()[0]
    benign = db.execute("SELECT COUNT(*) FROM patients WHERE result = 'Benign'").fetchone()[0]
    latest = db.execute("SELECT * FROM patients ORDER BY created_at DESC LIMIT 4").fetchall()
    return render_template(
        "dashboard.html",
        user=session["user"],
        total=total,
        malignant=malignant,
        benign=benign,
        latest=latest,
        model_mode=MODEL_MODE,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not login_required():
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            name = request.form.get("name", "").strip()
            age_raw = request.form.get("age", "").strip()
            uploaded = request.files.get("image")

            if not name or not age_raw:
                flash("Veuillez renseigner le nom et l'age du patient.", "warning")
                return redirect(url_for("predict"))
            if not uploaded or uploaded.filename == "":
                flash("Veuillez choisir une image.", "warning")
                return redirect(url_for("predict"))
            if not allowed_file(uploaded.filename):
                flash("Format non supporte. Utilisez JPG, JPEG, PNG ou WEBP.", "warning")
                return redirect(url_for("predict"))

            age = int(age_raw)
            if age < 1 or age > 120:
                flash("L'age doit etre compris entre 1 et 120 ans.", "warning")
                return redirect(url_for("predict"))

            upload_name = make_upload_name(uploaded.filename)
            upload_path = UPLOAD_FOLDER / upload_name
            uploaded.save(upload_path)
            image_quality = assess_image_quality(upload_path)
            if int(image_quality["score"]) < 55:
                flash("Image quality is low. Review lighting, focus, and resolution before trusting the scan.", "warning")

            img_arr = prepare_image(upload_path)
            probability = predict_probability(upload_path, img_arr)
            result = "Malignant" if probability >= 0.5 else "Benign"
            result_analysis = build_result_analysis(result, probability, image_quality)
            xai_paths = save_xai_images(upload_path, Path(upload_name).stem, img_arr)
            image_url = stored_to_url(upload_path)

            get_db().execute(
                """
                INSERT INTO patients
                    (name, age, result, probability, image_path, gradcam_path, heatmap_path, attention_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    age,
                    result,
                    probability,
                    image_url,
                    xai_paths["gradcam"],
                    xai_paths["heatmap"],
                    xai_paths["attention"],
                ),
            )
            get_db().commit()

            flash("Analyse terminee avec succes.", "success")
            return render_template(
                "result.html",
                name=name,
                age=age,
                result=result,
                prob=round(probability * 100, 2),
                img=image_url,
                gradcam=xai_paths["gradcam"],
                heatmap=xai_paths["heatmap"],
                attention=xai_paths["attention"],
                model_mode=MODEL_MODE,
                image_quality=image_quality,
                result_analysis=result_analysis,
            )
        except Exception as error:
            flash(f"Erreur systeme : {error}", "danger")
            return redirect(url_for("predict"))

    latest_scans = get_db().execute("SELECT * FROM patients ORDER BY created_at DESC LIMIT 3").fetchall()
    return render_template("predict.html", model_mode=MODEL_MODE, latest_scans=latest_scans)


@app.route("/patients")
def patients():
    if not login_required():
        return redirect(url_for("login"))

    rows = get_db().execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    patients_list = []
    for row in rows:
        item = dict(row)
        if isinstance(item.get("created_at"), str):
            try:
                item["created_at"] = datetime.strptime(item["created_at"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                item["created_at"] = None
        patients_list.append(item)
    return render_template("patients.html", patients=patients_list)


@app.route("/logout")
def logout():
    session.clear()
    flash("Vous avez ete deconnecte.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", use_reloader=False)
