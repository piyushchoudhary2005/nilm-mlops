"""
train_and_save.py
-----------------
Standalone training script called by the Jenkins pipeline (Stage 8).
Trains Random Forest, XGBoost, and SVR on UK-DALE data,
saves trained models as .pkl files, and saves evaluation plots.

Usage:
    python train_and_save.py --output-dir models --plots-dir outputs
"""

import os
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI
import matplotlib.pyplot as plt

import kagglehub
import h5py
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

SEED = 42
np.random.seed(SEED)

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="models",  help="Directory to save .pkl model files")
parser.add_argument("--plots-dir",  default="outputs", help="Directory to save evaluation plots")
parser.add_argument("--n-samples",  type=int, default=100_000, help="Number of samples to use")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.plots_dir,  exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
class Config:
    HOUSE_IDS      = [1, 2]
    SAMPLE_PERIOD  = 6
    APPLIANCES     = {"kettle": 10, "fridge": 5, "washing_machine": 6, "microwave": 13, "dish_washer": 22}
    WINDOW_SIZE    = 480
    TARGET         = "kettle"

cfg = Config()

# ── Download dataset ───────────────────────────────────────────────────────
print("📂 Downloading UK-DALE dataset...")
path = kagglehub.dataset_download("abdelmdz/uk-dale")
ukdale_path = os.path.join(path, "ukdale.h5")
print(f"✅ Dataset path: {ukdale_path}")

# ── Load data ──────────────────────────────────────────────────────────────
def load_ukdale(path):
    def _read_meter(store, building, meter):
        key = f"/building{building}/elec/meter{meter}"
        if key not in store:
            raise KeyError(f"Key {key} not found")
        df = store[key]
        for col in [("power", "active"), "power active", "active", 0]:
            if col in df.columns:
                return df[col].astype(np.float32)
        numeric = df.select_dtypes(include=[np.number]).columns
        if len(numeric):
            return df[numeric[0]].astype(np.float32)
        raise ValueError(f"No numeric column in {key}")

    all_agg, all_app = [], {n: [] for n in cfg.APPLIANCES}
    with pd.HDFStore(path, "r") as store:
        for house in cfg.HOUSE_IDS:
            try:
                agg    = _read_meter(store, house, 1)
                agg_t  = agg.index.astype(np.int64).values
                all_agg.append(agg.values.flatten())
                for app, chan in cfg.APPLIANCES.items():
                    try:
                        s = _read_meter(store, house, chan)
                        aligned = np.interp(agg_t, s.index.astype(np.int64).values,
                                            s.values.flatten().astype(np.float32))
                        all_app[app].append(np.clip(aligned, 0, None))
                    except Exception as e:
                        print(f"   ⚠️  Skipping {app} house {house}: {e}")
            except Exception as e:
                print(f"   ⚠️  Skipping house {house}: {e}")

    return {
        "aggregate":   np.concatenate(all_agg),
        "appliances":  {n: np.concatenate(v) for n, v in all_app.items() if v}
    }

raw = load_ukdale(ukdale_path)
N   = args.n_samples
times  = pd.date_range(start="2013-01-01", periods=len(raw["aggregate"]), freq=f"{cfg.SAMPLE_PERIOD}S")
agg_s  = pd.Series(raw["aggregate"],              index=times).head(N)
tgt_s  = pd.Series(raw["appliances"][cfg.TARGET],  index=times).head(N)
print(f"✅ Loaded {N:,} samples.")

# ── Feature engineering ────────────────────────────────────────────────────
def make_features(agg, cfg):
    ws   = int(cfg.WINDOW_SIZE * cfg.SAMPLE_PERIOD / 60 * 60 / cfg.SAMPLE_PERIOD)
    roll = agg.rolling(window=f"{ws}s", min_periods=1)
    df   = pd.DataFrame({
        "power_mean":   roll.mean(),
        "power_std":    roll.std(),
        "power_min":    roll.min(),
        "power_max":    roll.max(),
        "power_median": roll.median(),
        "power_skew":   roll.skew(),
        "power_kurt":   roll.kurt(),
        "power_rms":    np.sqrt(roll.apply(lambda x: np.mean(x**2))),
    }, index=agg.index).fillna(method="bfill").fillna(method="ffill")

    df["hour_of_day"]   = agg.index.hour
    df["day_of_week"]   = agg.index.dayofweek
    df["month_of_year"] = agg.index.month
    df["day_of_year"]   = agg.index.dayofyear
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]      = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    for lag in [1, 5, 10]:
        df[f"power_lag_{lag}"] = agg.shift(lag)
    df = df.fillna(method="bfill").fillna(method="ffill")
    df["power_diff"] = agg.diff().abs()
    df["is_event"]   = (df["power_diff"] > 500).astype(int)
    return df

print("🔧 Engineering features...")
feats  = make_features(agg_s, cfg).dropna()
y      = tgt_s.loc[feats.index]
X_train, X_test, y_train, y_test = train_test_split(feats, y, test_size=0.2, random_state=SEED)
print(f"   Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── Train & evaluate helper ────────────────────────────────────────────────
results = []

def train_eval(name, model, Xtr, ytr, Xte, yte, scale=False):
    scaler = None
    if scale:
        scaler = StandardScaler()
        Xtr    = scaler.fit_transform(Xtr)
        Xte    = scaler.transform(Xte)
    t0    = time.time()
    model.fit(Xtr, ytr)
    elapsed = time.time() - t0
    preds = model.predict(Xte)
    mae   = mean_absolute_error(yte, preds)
    rmse  = np.sqrt(mean_squared_error(yte, preds))
    r2    = r2_score(yte, preds)
    print(f"  ✅ {name:20s}  MAE={mae:.2f}W  RMSE={rmse:.2f}W  R²={r2:.4f}  ({elapsed:.1f}s)")

    # Save model
    artefact = {"model": model, "scaler": scaler, "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2}}
    pkl_path  = os.path.join(args.output_dir, f"{name.replace(' ', '_').replace('(','').replace(')','')}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artefact, f)
    print(f"     Saved → {pkl_path}")

    results.append({"Model": name, "MAE (W)": round(mae,2), "RMSE (W)": round(rmse,2),
                     "R²": round(r2,4), "Time (s)": round(elapsed,1), "predictions": preds})

# ── Random Forest ──────────────────────────────────────────────────────────
print("\n--- Training Random Forest ---")
train_eval("Random Forest",
           RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
           X_train, y_train, X_test, y_test)

# ── XGBoost ────────────────────────────────────────────────────────────────
print("\n--- Training XGBoost ---")
train_eval("XGBoost",
           xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=SEED, n_jobs=-1, verbosity=0),
           X_train, y_train, X_test, y_test)

# ── SVR ────────────────────────────────────────────────────────────────────
print("\n--- Training SVR ---")
SVR_MAX = 20_000
train_eval("SVR RBF",
           SVR(kernel="rbf", C=100, epsilon=10, gamma="scale"),
           X_train.iloc[:SVR_MAX], y_train.iloc[:SVR_MAX],
           X_test, y_test, scale=True)

# ── Comparison table ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("  MODEL COMPARISON")
print("="*55)
df = pd.DataFrame([{k:v for k,v in r.items() if k != "predictions"} for r in results])
df = df.sort_values("MAE (W)")
print(df.to_string(index=False))
print(f"\n🏆 Best model by MAE: {df.iloc[0]['Model']}")

# Save comparison CSV
csv_path = os.path.join(args.output_dir, "model_comparison.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Comparison saved → {csv_path}")

# ── Save comparison bar chart ──────────────────────────────────────────────
names  = df["Model"].tolist()
maes   = df["MAE (W)"].tolist()
rmses  = df["RMSE (W)"].tolist()
r2s    = df["R²"].tolist()
colors = ["#2196F3", "#4CAF50", "#FF9800"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Comparison — Kettle Power Disaggregation", fontsize=13, fontweight="bold")
for ax, vals, title, ylabel in zip(axes,
    [maes, rmses, r2s],
    ["MAE (W) — lower is better", "RMSE (W) — lower is better", "R² Score — higher is better"],
    ["MAE (W)", "RMSE (W)", "R²"]):
    bars = ax.bar(names, vals, color=colors)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
chart_path = os.path.join(args.plots_dir, "model_comparison.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Chart saved → {chart_path}")

print("\n✅ Training complete. All models and plots saved.")
