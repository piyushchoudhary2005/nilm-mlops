# -*- coding: utf-8 -*-
"""
UK-DALE NILM Mini Project — Multi-Model Comparison
=====================================================
Models compared:
  1. Random Forest Regressor  (original)
  2. XGBoost Regressor
  3. Support Vector Regressor (SVR)

Metrics: MAE, RMSE, R²
"""

# ─────────────────────────────────────────────
# 1.  INSTALL DEPENDENCIES
# ─────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("h5py")
install("pandas")
install("numpy")
install("scikit-learn")
install("matplotlib")
install("xgboost")
install("gradio")
install("kagglehub")

print("✅ All dependencies installed.")

# ─────────────────────────────────────────────
# 2.  IMPORTS
# ─────────────────────────────────────────────
import os, random, time
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import xgboost as xgb
import gradio as gr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 3.  CONFIG
# ─────────────────────────────────────────────
class Config:
    UKDALE_PATH    = None
    HOUSE_IDS      = [1, 2]
    SAMPLE_PERIOD  = 6

    APPLIANCES = {
        "kettle":           10,
        "fridge":            5,
        "washing_machine":   6,
        "microwave":        13,
        "dish_washer":      22,
    }
    ON_THRESHOLDS = {
        "kettle":           2000,
        "fridge":             50,
        "washing_machine":    20,
        "microwave":         200,
        "dish_washer":        10,
    }

    WINDOW_SIZE    = 480
    STRIDE         = 120
    CHECKPOINT_DIR = "/tmp/checkpoints"
    OUTPUT_DIR     = "/tmp"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

cfg = Config()

# ─────────────────────────────────────────────
# 4.  DOWNLOAD & LOAD DATA
# ─────────────────────────────────────────────
import kagglehub
os.makedirs("/usr/local/hdf5/lib/plugin", exist_ok=True)

path = kagglehub.dataset_download("abdelmdz/uk-dale")
cfg.UKDALE_PATH = os.path.join(path, "ukdale.h5")
print(f"✅ UK-DALE path: {cfg.UKDALE_PATH}")


def inspect_hdf5(path: str, max_depth: int = 3) -> None:
    def _walk(name, obj, depth=0):
        if depth > max_depth: return
        kind = "GROUP" if isinstance(obj, h5py.Group) else f"DATASET {obj.shape}"
        print("  " * depth + f"{name}  [{kind}]")
    print(f"\n↓️ HDF5 structure of {path}:")
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: _walk(name, obj, name.count("/")))

inspect_hdf5(cfg.UKDALE_PATH)


def load_ukdale(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ File not found: '{path}'")
    print(f"📂 Loading UK-DALE from houses {cfg.HOUSE_IDS}...")

    def _read_meter(store, building: int, meter: int) -> pd.Series:
        key = f"/building{building}/elec/meter{meter}"
        if key not in store:
            raise KeyError(f"Key {key} not in store")
        df = store[key]
        for col in [("power", "active"), "power active", "active", 0]:
            if col in df.columns:
                return df[col].astype(np.float32)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            return df[numeric_cols[0]].astype(np.float32)
        raise ValueError(f"No numeric column found in {key}.")

    all_agg_vals = []
    all_app_data = {name: [] for name in cfg.APPLIANCES}

    with pd.HDFStore(path, "r") as store:
        for house_id in cfg.HOUSE_IDS:
            try:
                agg_series = _read_meter(store, house_id, 1)
                agg_vals   = agg_series.values.flatten()
                agg_time   = agg_series.index.astype(np.int64).values
                all_agg_vals.append(agg_vals)
                print(f"   House {house_id} Aggregate: {len(agg_vals):,} pts | mean={agg_vals.mean():.1f}W")
                for app_name, chan in cfg.APPLIANCES.items():
                    try:
                        app_series = _read_meter(store, house_id, chan)
                        app_vals   = app_series.values.flatten().astype(np.float32)
                        app_time   = app_series.index.astype(np.int64).values
                        aligned    = np.interp(agg_time, app_time, app_vals)
                        all_app_data[app_name].append(np.clip(aligned, 0, None))
                        print(f"   ✅ House {house_id} {app_name:<15} {len(aligned):>10,} pts")
                    except Exception as e:
                        print(f"   ⚠️  House {house_id} Skipping {app_name}: {e}")
            except Exception as e:
                print(f"   ⚠️  Skipping House {house_id}: {e}")

    if not all_agg_vals:
        raise ValueError("❌ No aggregate data loaded.")

    full_agg_vals  = np.concatenate(all_agg_vals)
    full_appliance = {n: np.concatenate(v) for n, v in all_app_data.items() if v}
    return {"aggregate": full_agg_vals, "appliances": full_appliance}


raw_data = load_ukdale(cfg.UKDALE_PATH)
print("✅ raw_data loaded.")

# Build pandas Series
times = pd.date_range(start='2013-01-01', periods=len(raw_data["aggregate"]), freq=f'{cfg.SAMPLE_PERIOD}S')
aggregate_series      = pd.Series(raw_data["aggregate"], index=times, name='aggregate_power')

target_appliance_name  = 'kettle'
target_appliance_power = pd.Series(
    raw_data['appliances'][target_appliance_name],
    index=aggregate_series.index,
    name=f'{target_appliance_name}_power'
)

# Truncate for speed
N = 100_000
aggregate_series       = aggregate_series.head(N)
target_appliance_power = target_appliance_power.head(N)
print(f"Using {N:,} samples.")

# ─────────────────────────────────────────────
# 5.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def create_rf_features(aggregate_series: pd.Series, cfg) -> pd.DataFrame:
    """Statistical, temporal, lagged, and event-based features."""
    window_size_samples = int(cfg.WINDOW_SIZE * cfg.SAMPLE_PERIOD / 60 * 60 / cfg.SAMPLE_PERIOD)
    rolling = aggregate_series.rolling(window=f'{window_size_samples}s', min_periods=1)

    df = pd.DataFrame({
        'power_mean':   rolling.mean(),
        'power_std':    rolling.std(),
        'power_min':    rolling.min(),
        'power_max':    rolling.max(),
        'power_median': rolling.median(),
        'power_skew':   rolling.skew(),
        'power_kurt':   rolling.kurt(),
        'power_rms':    np.sqrt(rolling.apply(lambda x: np.mean(x**2))),
    }, index=aggregate_series.index)

    df = df.fillna(method='bfill').fillna(method='ffill')

    df['hour_of_day']   = aggregate_series.index.hour
    df['day_of_week']   = aggregate_series.index.dayofweek
    df['month_of_year'] = aggregate_series.index.month
    df['day_of_year']   = aggregate_series.index.dayofyear
    df['is_weekend']    = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin']      = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos']      = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    for lag in [1, 5, 10]:
        df[f'power_lag_{lag}'] = aggregate_series.shift(lag)
    df = df.fillna(method='bfill').fillna(method='ffill')

    df['power_diff'] = aggregate_series.diff().abs()
    df['is_event']   = (df['power_diff'] > 500).astype(int)
    return df

print("🔧 Engineering features …")
rf_features = create_rf_features(aggregate_series, cfg)
print(f"   Feature matrix: {rf_features.shape}")

X = rf_features.dropna()
y = target_appliance_power.loc[X.index]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"   Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 6.  HELPER — train & evaluate a model
# ─────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te, scale=False):
    print(f"\n{'─'*50}\n  Training: {name}")

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_pred = model.predict(X_te)
    mae    = mean_absolute_error(y_te, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    r2     = r2_score(y_te, y_pred)

    print(f"  ✅ {train_time:.1f}s  |  MAE={mae:.2f}W  RMSE={rmse:.2f}W  R²={r2:.4f}")
    return {"Model": name, "MAE (W)": round(mae, 2), "RMSE (W)": round(rmse, 2),
            "R²": round(r2, 4), "Train Time (s)": round(train_time, 1),
            "predictions": y_pred, "scaler": scaler}

# ─────────────────────────────────────────────
# 7.  TRAIN ALL THREE MODELS
# ─────────────────────────────────────────────
results = []

# ── 7a. Random Forest ─────────────────────────
rf_model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
results.append(evaluate("Random Forest", rf_model, X_train, y_train, X_test, y_test))

# ── 7b. XGBoost ───────────────────────────────
xgb_model = xgb.XGBRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1, verbosity=0
)
results.append(evaluate("XGBoost", xgb_model, X_train, y_train, X_test, y_test))

# ── 7c. SVR ───────────────────────────────────
# SVR is O(n²) so we cap training samples for speed
SVR_MAX   = 20_000
svr_model = SVR(kernel='rbf', C=100, epsilon=10, gamma='scale')
results.append(evaluate(
    "SVR (RBF)", svr_model,
    X_train.iloc[:SVR_MAX], y_train.iloc[:SVR_MAX],
    X_test, y_test,
    scale=True
))

# ─────────────────────────────────────────────
# 8.  COMPARISON TABLE
# ─────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  MODEL COMPARISON — {target_appliance_name.upper()}")
print(f"{'═'*55}")

summary = [{k: v for k, v in r.items() if k not in ("predictions", "scaler")} for r in results]
df_summary = pd.DataFrame(summary).sort_values("MAE (W)")
print(df_summary.to_string(index=False))

best_model_name = df_summary.iloc[0]["Model"]
print(f"\n🏆 Best model by MAE: {best_model_name}")

# ─────────────────────────────────────────────
# 9.  COMPARISON BAR CHARTS
# ─────────────────────────────────────────────
model_names = df_summary["Model"].tolist()
maes        = df_summary["MAE (W)"].tolist()
rmses       = df_summary["RMSE (W)"].tolist()
r2s         = df_summary["R²"].tolist()
colors      = ['#2196F3', '#4CAF50', '#FF9800']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Model Comparison — {target_appliance_name.capitalize()} Power Disaggregation",
             fontsize=13, fontweight='bold')

for ax, vals, title, ylabel in zip(
    axes,
    [maes, rmses, r2s],
    ["MAE (W) — lower is better", "RMSE (W) — lower is better", "R² Score — higher is better"],
    ["MAE (W)", "RMSE (W)", "R²"]
):
    bars = ax.bar(model_names, vals, color=colors)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("/tmp/model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Comparison chart saved.")

# ─────────────────────────────────────────────
# 10. PREDICTION PLOTS — all 3 models
# ─────────────────────────────────────────────
SEG = 500
fig2, axes2 = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
fig2.suptitle(f"Predicted vs Actual — {target_appliance_name.capitalize()} (first {SEG} test samples)",
              fontsize=12, fontweight='bold')

for ax, res in zip(axes2, results):
    actual = y_test.values[:SEG]
    preds  = res["predictions"][:SEG]
    ax.plot(actual, label='Actual',    alpha=0.85, linewidth=1)
    ax.plot(preds,  label='Predicted', alpha=0.85, linewidth=1, linestyle='--')
    ax.set_title(f"{res['Model']}   MAE={res['MAE (W)']}W  |  RMSE={res['RMSE (W)']}W  |  R²={res['R²']}",
                 fontsize=10)
    ax.set_ylabel("Power (W)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("/tmp/prediction_plots.png", dpi=120, bbox_inches='tight')
plt.show()
print("✅ Prediction plots saved.")

# ─────────────────────────────────────────────
# 11. SCATTER — best model actual vs predicted
# ─────────────────────────────────────────────
best_res  = next(r for r in results if r["Model"] == best_model_name)
y_act_b   = y_test.values
y_pred_b  = best_res["predictions"]

plt.figure(figsize=(7, 7))
plt.scatter(y_act_b, y_pred_b, alpha=0.2, s=5, color='steelblue')
lim = [min(y_act_b.min(), y_pred_b.min()), max(y_act_b.max(), y_pred_b.max())]
plt.plot(lim, lim, '--r', linewidth=2, label='Perfect Prediction')
plt.title(f'Best Model: {best_model_name}\nActual vs Predicted — {target_appliance_name.capitalize()}')
plt.xlabel('Actual Power (W)')
plt.ylabel('Predicted Power (W)')
plt.legend(); plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("/tmp/best_model_scatter.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 12. GRADIO DEMO
# ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "Random Forest": (rf_model,  results[0]["scaler"]),
    "XGBoost":       (xgb_model, results[1]["scaler"]),
    "SVR (RBF)":     (svr_model, results[2]["scaler"]),
}

def gradio_predict(model_name: str, start_offset_minutes: int, duration_minutes: int):
    model_obj, scaler = MODEL_REGISTRY[model_name]
    spm          = 60 / cfg.SAMPLE_PERIOD
    start_sample = int(start_offset_minutes * spm)
    end_sample   = min(int(start_sample + duration_minutes * spm), len(aggregate_series))
    start_sample = max(0, min(start_sample, end_sample - int(spm)))

    agg_seg    = aggregate_series.iloc[start_sample:end_sample]
    target_seg = target_appliance_power.iloc[start_sample:end_sample]

    feats = create_rf_features(agg_seg, cfg).dropna()
    X_seg = feats.values
    y_act = target_seg.loc[feats.index].values

    if scaler:
        X_seg = scaler.transform(X_seg)

    y_prd    = model_obj.predict(X_seg)
    mae_seg  = mean_absolute_error(y_act, y_prd)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_act, label='Actual',    alpha=0.85)
    ax.plot(y_prd, label='Predicted', linestyle='--', alpha=0.85)
    ax.set_title(f'{model_name} — {target_appliance_name.capitalize()}  |  Segment MAE: {mae_seg:.1f}W')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Power (W)')
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    return fig

demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(choices=list(MODEL_REGISTRY.keys()), value="Random Forest", label="Model"),
        gr.Slider(0, 10000, value=0,   step=10, label="Start Offset (minutes)"),
        gr.Slider(10, 1440, value=120, step=10, label="Duration (minutes)"),
    ],
    outputs=gr.Plot(label="Disaggregation Result"),
    title="UK-DALE NILM — 3-Model Disaggregation Demo",
    description=f"Compare Random Forest, XGBoost, and SVR on {target_appliance_name} power disaggregation.",
)

demo.launch(share=True)
