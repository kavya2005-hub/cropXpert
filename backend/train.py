"""
CropXpert — Model Trainer
=========================
Run this script ONCE (from inside the backend/ directory) to generate:
  model/crop_model.pkl   — trained RandomForestClassifier
  model/encoders.pkl     — LabelEncoders for all categorical columns

Usage:
  cd backend
  python train.py

The saved model and encoders are loaded by app.py at startup.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# ============================================================
# 1. Load Dataset
# ============================================================

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "dataset", "dataset.csv")

print(f"[Train] Loading dataset from: {DATASET_PATH}")
data = pd.read_csv(DATASET_PATH)

# Normalise column names — remove accidental whitespace
data.columns = data.columns.str.strip()

# Fix inconsistent column name
if "Land Size" in data.columns:
    data.rename(columns={"Land Size": "LandSize"}, inplace=True)

# Strip whitespace from all string columns
str_cols = ["Climate", "SoilType", "WaterLevel",
            "PlantName", "FertilizerName", "DiseasesName", "MedicineName"]
for col in str_cols:
    if col in data.columns:
        data[col] = data[col].astype(str).str.strip()

print(f"[Train] Dataset shape: {data.shape}")
print(f"[Train] Unique plants : {data['PlantName'].nunique()}")

# ============================================================
# 2. Label-Encode All Categorical Columns
# ============================================================
# WHY encode everything?
#   RandomForest requires numeric input.
#   LabelEncoder converts each unique string to a stable integer.
#   We save the encoders so app.py can:
#     a) encode incoming user text the same way
#     b) decode the model's integer output back to a plant name

encoders = {}

categorical_cols = [
    "Climate", "SoilType", "WaterLevel",
    "PlantName", "FertilizerName", "DiseasesName", "MedicineName"
]

for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
        print(f"[Encode] {col}: {len(le.classes_)} unique values")
    else:
        print(f"[Encode] WARNING: column '{col}' not found in dataset — skipping")

# ============================================================
# 3. Define Features and Target
# ============================================================
# Features (X): what the farmer tells us
#   - Climate (encoded int)
#   - SoilType (encoded int)
#   - WaterLevel (encoded int)
#   - LandSize (raw float — already numeric, no encoding needed)
#
# Target (y): what we want to predict
#   - PlantName (encoded int — decoded back to string in app.py)
# ============================================================

required_cols = ["Climate", "SoilType", "WaterLevel", "LandSize", "PlantName"]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    raise Exception(f"[Train] ABORT — missing required columns: {missing}")

X = data[["Climate", "SoilType", "WaterLevel", "LandSize"]]
y = data["PlantName"]   # at this point PlantName is already encoded to integers

print(f"\n[Train] Feature shape: {X.shape}")
print(f"[Train] Target classes: {y.nunique()}")

# ============================================================
# 4. Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.value_counts().min() >= 2 else None
)

print(f"\n[Train] Train samples: {len(X_train)}")
print(f"[Train] Test  samples: {len(X_test)}")

# ============================================================
# 5. Train RandomForestClassifier
# ============================================================
# n_estimators=100  — more trees = more stable predictions (vs 25 before)
# max_depth=12      — deeper trees capture more patterns
# random_state=42   — reproducible results
# n_jobs=-1         — use all CPU cores for faster training

print("\n[Train] Training RandomForestClassifier ...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ============================================================
# 6. Evaluate
# ============================================================

y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
print(f"[Train] Test Accuracy: {accuracy * 100:.2f}%")

# ============================================================
# 7. Save Model and Encoders
# ============================================================

MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "crop_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(encoders, f)

print(f"\n[Train] ✅ Model saved    → {MODEL_PATH}")
print(f"[Train] ✅ Encoders saved → {ENCODER_PATH}")
print("\n[Train] Done. You can now start app.py.")
