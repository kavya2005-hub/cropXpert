"""
CropXpert AI — Flask Backend
=============================
Phase 1 : Top-3 AI crop recommendations with confidence scores + reasons
Phase 2 : Weather automation via OpenWeatherMap API (removes manual climate)
Phase 3 : Automated recommendation history saved to MySQL
Phase 4 : Automated PDF report generation and download
Phase 5 : Architecture stubs for CNN / chatbot / mobile future extensions
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from fpdf import FPDF
import mysql.connector
import mysql.connector.pooling
import pandas as pd
import numpy as np
import pickle
import requests
import os
import io
import datetime

# ============================================================
# ENV + APP INIT
# ============================================================

load_dotenv()
app = Flask(__name__)
CORS(app)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "model",   "crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model",   "encoders.pkl")

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# ============================================================
# DATABASE POOL
# ============================================================

db_pool = None

def init_db_pool():
    global db_pool
    try:
        db_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="cropxpert_pool",
            pool_size=5,
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "cropxpert"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        print("[DB] Pool created.")
    except Exception as e:
        db_pool = None
        print(f"[DB] Pool failed: {e}")

def get_db():
    if db_pool is None:
        return None
    try:
        return db_pool.get_connection()
    except Exception as e:
        print(f"[DB] Get connection failed: {e}")
        return None

init_db_pool()

# ============================================================
# DATASET — loaded once, reused for every request
# ============================================================

crop_data = None

try:
    crop_data = pd.read_csv(DATASET_PATH)
    crop_data.columns = crop_data.columns.str.strip()
    if "Land Size" in crop_data.columns:
        crop_data.rename(columns={"Land Size": "LandSize"}, inplace=True)
    for col in ["Climate","SoilType","WaterLevel","PlantName",
                "FertilizerName","DiseasesName","MedicineName"]:
        if col in crop_data.columns:
            crop_data[col] = crop_data[col].astype(str).str.strip()
    print(f"[Dataset] {len(crop_data)} rows loaded.")
except Exception as e:
    crop_data = None
    print(f"[Dataset] Load failed: {e}")

# ============================================================
# ML MODEL + ENCODERS — loaded once at startup
# ============================================================

crop_model   = None
ml_encoders  = None
ml_available = False

def load_ml_assets():
    global crop_model, ml_encoders, ml_available
    try:
        with open(MODEL_PATH, "rb") as f:
            crop_model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            ml_encoders = pickle.load(f)
        ml_available = True
        print(f"[ML] Model loaded — {len(crop_model.classes_)} classes.")
    except FileNotFoundError as e:
        print(f"[ML] File not found: {e}. Run train.py first.")
    except Exception as e:
        print(f"[ML] Load failed: {e}")

load_ml_assets()

# ============================================================
# HELPERS
# ============================================================

def normalise_water(raw: str) -> str:
    """Map any casing of water level to Title-Case for the encoder."""
    return {"low": "Low", "medium": "Medium", "high": "High"}.get(
        raw.strip().lower(), raw.strip()
    )

# Maps every possible frontend climate value → exact encoder class
# Protects against future UI changes causing 400 errors
_CLIMATE_MAP = {
    "summer":        "Summer",
    "winter":        "Winter",
    "spring":        "Spring",
    "cold":          "Cold",
    "rainy":         "Rainy (Monsoon)",      # frontend used to send "Rainy"
    "rainy (monsoon)": "Rainy (Monsoon)",    # correct full value
    "monsoon":       "Rainy (Monsoon)",
    "rain":          "Rainy (Monsoon)",
}

def normalise_climate(raw: str) -> str:
    """
    Map any incoming climate string to the exact label the encoder knows.
    Case-insensitive.  Returns the original string unchanged if not in the map,
    so the encoder's own ValueError will fire with a useful message.
    """
    return _CLIMATE_MAP.get(raw.strip().lower(), raw.strip())

def encode_input(climate, soil, water, land):
    """
    Encode 4 user inputs to integers the RandomForest expects.
    Returns (DataFrame, None) on success or (None, error_string) on failure.
    """
    try:
        c = ml_encoders["Climate"].transform([climate])[0]
    except ValueError:
        return None, f"Invalid climate '{climate}'. Valid: {list(ml_encoders['Climate'].classes_)}"
    try:
        s = ml_encoders["SoilType"].transform([soil])[0]
    except ValueError:
        return None, f"Invalid soil_type '{soil}'. Valid: {list(ml_encoders['SoilType'].classes_)}"
    try:
        w = ml_encoders["WaterLevel"].transform([water])[0]
    except ValueError:
        return None, f"Invalid water_level '{water}'. Valid: {list(ml_encoders['WaterLevel'].classes_)}"
    X = pd.DataFrame([[c, s, w, land]],
                     columns=["Climate","SoilType","WaterLevel","LandSize"])
    return X, None

def get_crop_details(plant_name: str) -> dict:
    """Look up fertilizer, disease, medicine for a plant from the CSV."""
    default = {"fertilizer_name": "N/A", "disease_name": "N/A", "medicine_name": "N/A"}
    if crop_data is None:
        return default
    matches = crop_data[crop_data["PlantName"].str.lower() == plant_name.lower()]
    if matches.empty:
        return default
    row = matches.iloc[0]
    return {
        "fertilizer_name": str(row.get("FertilizerName", "N/A")).strip(),
        "disease_name":    str(row.get("DiseasesName",   "N/A")).strip(),
        "medicine_name":   str(row.get("MedicineName",   "N/A")).strip()
    }

def build_reasons(climate: str, soil: str, water: str) -> list:
    """
    Generate human-readable explanation sentences for why a crop was recommended.
    These are rule-based strings derived from the input features — they explain
    the AI's decision in plain language.
    """
    water_map = {"low": "low water", "medium": "moderate water", "high": "high water"}
    water_label = water_map.get(water.lower(), water)
    return [
        f"Suitable for {soil} soil",
        f"Requires {water_label}",
        f"Performs well in {climate} climate"
    ]

def save_recommendation_to_db(user_id, city, climate, soil, water,
                               land, crop_name, confidence,
                               fertilizer, disease, medicine,
                               temperature=None, humidity=None):
    """
    Phase 3 — persist every recommendation to the `recommendations` table.
    Called automatically after every successful prediction.
    Silently skips if DB is unavailable so the API response is never blocked.
    """
    conn = cursor = None
    try:
        conn = get_db()
        if conn is None:
            return
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO recommendations
              (user_id, city, climate, soil_type, water_level, land_size,
               crop_name, confidence, fertilizer, disease, medicine,
               temperature, humidity)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (user_id, city, climate, soil, water, land,
              crop_name, confidence, fertilizer, disease, medicine,
              temperature, humidity))
        conn.commit()
    except Exception as e:
        print(f"[History] Save failed: {e}")
    finally:
        if cursor: cursor.close()
        if conn:   conn.close()

# ============================================================
# PHASE 2 — WEATHER AUTOMATION HELPER
# ============================================================
# Maps temperature + weather condition → one of the 5 climate labels
# the RandomForest was trained on: Cold, Winter, Spring, Summer, Rainy

def derive_climate(temp_c: float, description: str) -> str:
    """
    AI logic to convert raw weather data into a crop-model climate label.

    Rules (in priority order):
      1. Any rain/drizzle/thunderstorm keyword → 'Rainy (Monsoon)'
      2. Temperature < 10°C                   → 'Cold'
      3. Temperature 10–18°C                  → 'Winter'
      4. Temperature 18–28°C                  → 'Spring'
      5. Temperature > 28°C                   → 'Summer'
    """
    desc_lower = description.lower()
    rain_keywords = ["rain", "drizzle", "thunderstorm", "shower", "storm"]
    if any(kw in desc_lower for kw in rain_keywords):
        return "Rainy (Monsoon)"
    if temp_c < 10:
        return "Cold"
    if temp_c < 18:
        return "Winter"
    if temp_c < 28:
        return "Spring"
    return "Summer"

def fetch_weather(city: str) -> dict:
    """
    Call OpenWeatherMap Current Weather API and return structured data.

    Returns:
        {
          "temperature": 32.1,       # Celsius
          "humidity": 65,            # percentage
          "description": "clear sky",
          "climate": "Summer"        # derived label for the ML model
        }
    Or raises ValueError with a user-friendly message on failure.
    """
    if not OPENWEATHER_KEY or OPENWEATHER_KEY == "your_api_key_here":
        raise ValueError(
            "OpenWeather API key not configured. "
            "Set OPENWEATHER_API_KEY in backend/.env"
        )

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q":     city,
        "appid": OPENWEATHER_KEY,
        "units": "metric"   # Celsius
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
    except requests.exceptions.Timeout:
        raise ValueError("Weather service timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Cannot reach weather service. Check your internet connection.")

    if resp.status_code == 401:
        raise ValueError("Invalid OpenWeather API key. Check OPENWEATHER_API_KEY in .env")
    if resp.status_code == 404:
        raise ValueError(f"City '{city}' not found. Please check the spelling.")
    if resp.status_code != 200:
        raise ValueError(f"Weather API error (HTTP {resp.status_code}).")

    data = resp.json()
    temp        = float(data["main"]["temp"])
    humidity    = float(data["main"]["humidity"])
    description = data["weather"][0]["description"]
    climate     = derive_climate(temp, description)

    return {
        "temperature": round(temp, 1),
        "humidity":    round(humidity, 1),
        "description": description,
        "climate":     climate
    }

# ============================================================
# PHASE 4 — PDF GENERATOR
# ============================================================

class CropXpertPDF(FPDF):
    """Custom FPDF subclass with a branded header and footer."""

    def header(self):
        self.set_fill_color(255, 140, 0)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.cell(0, 18, "  CropXpert AI - Crop Recommendation Report", ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 100, 30)
        self.cell(0, 10,
                  f"Generated by CropXpert AI  |  Page {self.page_no()}  |  "
                  f"{datetime.datetime.now().strftime('%d %b %Y %H:%M')}",
                  align="C")


def generate_pdf(farmer_name, city, climate, soil, water, land,
                 weather_info, recommendations) -> bytes:
    """
    Build the PDF in memory and return raw bytes.
    Uses fpdf2 — no disk writes needed, safe for concurrent requests.
    """
    pdf = CropXpertPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Farmer Info Section ────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(255, 243, 224)
    pdf.cell(0, 8, "Farmer Information", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)

    info_rows = [
        ("Farmer Name",  farmer_name or "N/A"),
        ("Location",     city or "N/A"),
        ("Soil Type",    soil),
        ("Water Level",  water),
        ("Land Size",    f"{land} Acres"),
        ("Report Date",  datetime.datetime.now().strftime("%d %B %Y, %H:%M")),
    ]
    for label, value in info_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(50, 7, label + ":", border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, str(value), ln=True)
    pdf.ln(4)

    # ── Weather Section ────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(255, 243, 224)
    pdf.cell(0, 8, "Weather Information", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)

    if weather_info:
        weather_rows = [
            ("Climate Label",  weather_info.get("climate", climate)),
            ("Temperature",    f"{weather_info.get('temperature', 'N/A')} C"),
            ("Humidity",       f"{weather_info.get('humidity', 'N/A')} %"),
            ("Condition",      weather_info.get("description", "N/A").title()),
        ]
    else:
        weather_rows = [("Climate (manual)", climate)]

    for label, value in weather_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(50, 7, label + ":", border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, str(value), ln=True)
    pdf.ln(6)

    # ── Recommendations Section ────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(255, 243, 224)
    pdf.cell(0, 8, "AI Crop Recommendations", ln=True, fill=True)
    pdf.ln(3)

    colors = [(76, 175, 80), (33, 150, 243), (255, 152, 0)]

    for i, rec in enumerate(recommendations):
        r, g, b = colors[i % len(colors)]
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        conf = rec.get("confidence", 0)
        pdf.cell(0, 8,
                 f"  #{i+1}  {rec['plant_name']}   (Confidence: {conf:.1f}%)",
                 ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(1)

        detail_rows = [
            ("Fertilizer",  rec.get("fertilizer_name", "N/A")),
            ("Disease Risk",rec.get("disease_name",    "N/A")),
            ("Treatment",   rec.get("medicine_name",   "N/A")),
        ]
        for label, value in detail_rows:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(40, 6, "   " + label + ":", border=0)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 6, str(value), ln=True)

        reasons = rec.get("reason", [])
        if reasons:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 5, "   Why: " + "  |  ".join(reasons), ln=True)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    return bytes(pdf.output())

# ============================================================
# ROUTES — Auth (unchanged API contract)
# ============================================================

@app.route("/signup", methods=["POST"])
def signup():
    conn = cursor = None
    try:
        req = request.get_json()
        if not req:
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400
        name     = str(req.get("name",     "") or "").strip()
        phone    = str(req.get("phone",    "") or "").strip()
        email    = str(req.get("email",    "") or "").strip()
        gender   = str(req.get("gender",   "") or "").strip()
        city     = str(req.get("city",     "") or "").strip()
        password = str(req.get("password", "") or "").strip()
        if not name:
            return jsonify({"status": "error", "message": "Name is required"}), 400
        if not email or "@" not in email:
            return jsonify({"status": "error", "message": "Valid email is required"}), 400
        if not password or len(password) < 6:
            return jsonify({"status": "error", "message": "Password must be at least 6 characters"}), 400
        if not phone:
            return jsonify({"status": "error", "message": "Phone is required"}), 400
        conn = get_db()
        if conn is None:
            return jsonify({"status": "error", "message": "Database unavailable"}), 503
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name,phone,email,gender,city,password) VALUES (%s,%s,%s,%s,%s,%s)",
            (name, phone, email, gender, city, generate_password_hash(password))
        )
        conn.commit()
        return jsonify({"status": "success", "message": "Account created successfully"})
    except mysql.connector.IntegrityError:
        return jsonify({"status": "error", "message": "Email already registered"}), 409
    except Exception:
        return jsonify({"status": "error", "message": "Signup failed. Try again."}), 500
    finally:
        if cursor: cursor.close()
        if conn:   conn.close()


@app.route("/login", methods=["POST"])
def login():
    conn = cursor = None
    try:
        req = request.get_json()
        if not req:
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400
        email    = str(req.get("email",    "") or "").strip()
        password = str(req.get("password", "") or "").strip()
        if not email or not password:
            return jsonify({"status": "error", "message": "Email and password required"}), 400
        conn = get_db()
        if conn is None:
            return jsonify({"status": "error", "message": "Database unavailable"}), 503
        cursor = conn.cursor()
        cursor.execute("SELECT id, password, name FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        if user and check_password_hash(user[1], password):
            return jsonify({"status": "success", "user_id": user[0], "name": user[2], "message": "Login success"})
        return jsonify({"status": "error", "message": "Invalid email or password"}), 401
    except Exception:
        return jsonify({"status": "error", "message": "Login failed. Try again."}), 500
    finally:
        if cursor: cursor.close()
        if conn:   conn.close()

# ============================================================
# PHASE 1 — /crop-recommendation  (Top-3 with confidence + reasons)
# PHASE 2 — Weather automation via city input
# PHASE 3 — Auto-save to recommendations table
# ============================================================

@app.route("/crop-recommendation", methods=["POST"])
def predict_crop():
    try:
        req = request.get_json()
        if not req:
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        # ── Debug log — visible in Flask terminal ─────────────
        print(f"[/crop-recommendation] Received: {req}")

        # ── Read inputs ───────────────────────────────────────
        soil_raw  = str(req.get("soil_type",   "") or "").strip()
        water_raw = str(req.get("water_level", "") or "").strip()
        city      = str(req.get("city",        "") or "").strip()
        user_id   = req.get("user_id")

        try:
            land = float(req.get("land_size", 0) or 0)
            if land < 0:
                raise ValueError()
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "land_size must be a positive number"}), 400

        if not soil_raw or not water_raw:
            return jsonify({"status": "error", "message": "soil_type and water_level are required"}), 400

        # ── Phase 2: weather fetch or manual climate ──────────
        weather_info = None
        temperature  = None
        humidity_val = None

        if city:
            try:
                weather_info = fetch_weather(city)
                climate_raw  = weather_info["climate"]
                temperature  = weather_info["temperature"]
                humidity_val = weather_info["humidity"]
            except ValueError as e:
                return jsonify({"status": "error", "message": str(e)}), 400
        else:
            # No city — use manually supplied climate (backward-compatible)
            climate_raw = str(req.get("climate", "") or "").strip()
            if not climate_raw:
                return jsonify({
                    "status":  "error",
                    "message": "Provide either 'city' for auto-weather or 'climate' manually"
                }), 400
            # Normalise to exact encoder label (fixes "Rainy" → "Rainy (Monsoon)")
            climate_raw = normalise_climate(climate_raw)

        water_normalised = normalise_water(water_raw)

        if not ml_available:
            return jsonify({"status": "error", "message": "ML model unavailable. Run train.py."}), 503

        # ── Phase 1: encode → predict_proba → Top-3 ──────────
        X, err = encode_input(climate_raw, soil_raw, water_normalised, land)
        if err:
            return jsonify({"status": "error", "message": err}), 400

        # predict_proba returns shape (1, n_classes) — probabilities per plant
        proba       = crop_model.predict_proba(X)[0]
        # model.classes_ = [0, 1, 2, ..., 495]  (integer codes)
        top3_idx    = np.argsort(proba)[::-1][:3]

        recommendations = []
        for rank, idx in enumerate(top3_idx):
            code       = crop_model.classes_[idx]
            confidence = round(float(proba[idx]) * 100, 2)
            plant_name = str(ml_encoders["PlantName"].inverse_transform([code])[0])
            details    = get_crop_details(plant_name)
            reasons    = build_reasons(climate_raw, soil_raw, water_raw)

            rec = {
                "plant_name":      plant_name,
                "confidence":      confidence,
                "fertilizer_name": details["fertilizer_name"],
                "disease_name":    details["disease_name"],
                "medicine_name":   details["medicine_name"],
                "reason":          reasons
            }
            recommendations.append(rec)

            # Phase 3 — save top recommendation (rank 0) to history
            if rank == 0 and user_id:
                save_recommendation_to_db(
                    user_id, city, climate_raw, soil_raw, water_raw, land,
                    plant_name, confidence,
                    details["fertilizer_name"],
                    details["disease_name"],
                    details["medicine_name"],
                    temperature, humidity_val
                )

        response = {"status": "success", "recommendations": recommendations}
        if weather_info:
            response["weather"] = weather_info   # include weather data in response

        return jsonify(response)

    except Exception as e:
        print(f"[Recommendation] Error: {e}")
        return jsonify({"status": "error", "message": "Recommendation failed. Try again."}), 500

# ============================================================
# PHASE 3 — GET /recommendation-history
# ============================================================

@app.route("/recommendation-history", methods=["GET"])
def recommendation_history():
    """
    Returns all past recommendations for the logged-in user plus statistics:
      - total_recommendations
      - most_recommended_crop
      - last_recommendation_date
    Query param: ?user_id=123
    """
    conn = cursor = None
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"status": "error", "message": "user_id required"}), 400

        conn = get_db()
        if conn is None:
            return jsonify({"status": "error", "message": "Database unavailable"}), 503

        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, city, climate, soil_type, water_level, land_size,
                   crop_name, confidence, fertilizer, disease, medicine,
                   temperature, humidity, created_at
            FROM recommendations
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        rows = cursor.fetchall()

        # Convert datetime objects to strings for JSON serialisation
        for row in rows:
            if row.get("created_at"):
                row["created_at"] = row["created_at"].strftime("%d %b %Y, %H:%M")

        # Stats
        total = len(rows)
        most_common = None
        last_date   = rows[0]["created_at"] if rows else None

        if rows:
            from collections import Counter
            counts      = Counter(r["crop_name"] for r in rows)
            most_common = counts.most_common(1)[0][0]

        return jsonify({
            "status":  "success",
            "history": rows,
            "stats": {
                "total_recommendations":  total,
                "most_recommended_crop":  most_common,
                "last_recommendation_date": last_date
            }
        })

    except Exception as e:
        print(f"[History] Error: {e}")
        return jsonify({"status": "error", "message": "Could not fetch history."}), 500
    finally:
        if cursor: cursor.close()
        if conn:   conn.close()


# ============================================================
# PHASE 4 — POST /generate-pdf
# ============================================================

@app.route("/generate-pdf", methods=["POST"])
def generate_pdf_report():
    """
    Accepts the recommendation payload and returns a downloadable PDF.
    The PDF is built entirely in RAM — no temp files on disk.

    Request JSON:
      {
        "farmer_name":    "Kavya",
        "city":           "Chennai",
        "climate":        "Summer",        # used if no weather_info
        "soil_type":      "Loamy",
        "water_level":    "Low",
        "land_size":      2.5,
        "weather_info":   { ... },         # optional — from /crop-recommendation response
        "recommendations":[ ... ]          # required — list of recommendation objects
      }
    """
    try:
        req = request.get_json()
        if not req:
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        farmer_name  = str(req.get("farmer_name",  "Farmer") or "Farmer").strip()
        city         = str(req.get("city",         "")       or "").strip()
        climate      = str(req.get("climate",      "")       or "").strip()
        soil         = str(req.get("soil_type",    "")       or "").strip()
        water        = str(req.get("water_level",  "")       or "").strip()
        weather_info = req.get("weather_info")
        recs         = req.get("recommendations", [])

        try:
            land = float(req.get("land_size", 0) or 0)
        except (ValueError, TypeError):
            land = 0.0

        if not recs:
            return jsonify({"status": "error",
                            "message": "recommendations list is required"}), 400

        pdf_bytes = generate_pdf(
            farmer_name, city, climate, soil, water, land,
            weather_info, recs
        )

        filename = f"CropXpert_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        print(f"[PDF] Error: {e}")
        return jsonify({"status": "error", "message": "PDF generation failed."}), 500


# ============================================================
# HEALTH CHECK — GET /health
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    """
    Quick diagnostic endpoint.
    Open http://127.0.0.1:5000/health in your browser to confirm
    the server, ML model, dataset, and DB pool all loaded correctly.
    """
    return jsonify({
        "status":   "ok",
        "ml_model": "loaded" if ml_available else "NOT loaded — run: python train.py",
        "dataset":  f"{len(crop_data)} rows" if crop_data is not None else "NOT loaded",
        "db_pool":  "connected" if db_pool is not None else "NOT connected — check .env"
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
