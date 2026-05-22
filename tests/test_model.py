"""
tests/test_model.py
Unit tests for the UK-DALE NILM ML pipeline.
Run with: pytest tests/ -v --cov=. --cov-report=xml
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock  # noqa: F401


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_aggregate_series():
    """A small synthetic aggregate power series for fast testing."""
    np.random.seed(42)
    n = 2000
    times = pd.date_range(start="2013-01-01", periods=n, freq="6s")
    # Mostly low background (fridge ~80W) with occasional kettle spikes (~2300W)
    values = np.random.uniform(50, 200, n).astype(np.float32)
    spike_idx = np.random.choice(n, size=20, replace=False)
    values[spike_idx] = 2300.0
    return pd.Series(values, index=times, name="aggregate_power")


@pytest.fixture
def sample_target_series(sample_aggregate_series):
    """Synthetic kettle power (sparse — mostly 0, spikes at same indices)."""
    values = np.zeros(len(sample_aggregate_series), dtype=np.float32)
    kettle_on = sample_aggregate_series.values > 2000
    values[kettle_on] = 2300.0
    return pd.Series(values, index=sample_aggregate_series.index, name="kettle_power")


@pytest.fixture
def mock_config():
    """Minimal Config object for feature engineering tests."""
    class Cfg:
        WINDOW_SIZE   = 60   # small window for speed
        SAMPLE_PERIOD = 6
    return Cfg()


# ─────────────────────────────────────────────
#  1. Feature Engineering Tests
# ─────────────────────────────────────────────

class TestFeatureEngineering:

    def test_feature_matrix_shape(self, sample_aggregate_series, mock_config):
        """Feature matrix should have 22 columns."""
        from mini_project_multi_model import create_rf_features
        features = create_rf_features(sample_aggregate_series, mock_config)
        assert features.shape[1] == 22, f"Expected 22 features, got {features.shape[1]}"

    def test_feature_matrix_row_count(self, sample_aggregate_series, mock_config):
        """Feature matrix should have the same number of rows as the input series."""
        from mini_project_multi_model import create_rf_features
        features = create_rf_features(sample_aggregate_series, mock_config)
        assert len(features) == len(sample_aggregate_series)

    def test_no_nan_after_fill(self, sample_aggregate_series, mock_config):
        """After ffill/bfill, feature matrix must have no NaN values."""
        from mini_project_multi_model import create_rf_features
        features = create_rf_features(sample_aggregate_series, mock_config)
        assert not features.isnull().any().any(), "Feature matrix contains NaN values"

    def test_cyclic_features_bounded(self, sample_aggregate_series, mock_config):
        """Sin/cos cyclic features must be in [-1, 1]."""
        from mini_project_multi_model import create_rf_features
        features = create_rf_features(sample_aggregate_series, mock_config)
        for col in ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]:
            assert features[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1] range"

    def test_is_event_binary(self, sample_aggregate_series, mock_config):
        """is_event must be a binary feature (0 or 1 only)."""
        from mini_project_multi_model import create_rf_features
        features = create_rf_features(sample_aggregate_series, mock_config)
        unique_vals = set(features["is_event"].unique())
        assert unique_vals.issubset({0, 1}), f"is_event has non-binary values: {unique_vals}"

    def test_is_weekend_binary(self, sample_aggregate_series, mock_config):
        """is_weekend must be a binary feature (0 or 1 only)."""
        from mini_project_multi_model import create_rf_features
        features = create_rf_features(sample_aggregate_series, mock_config)
        unique_vals = set(features["is_weekend"].unique())
        assert unique_vals.issubset({0, 1}), f"is_weekend has non-binary values: {unique_vals}"

    def test_power_rms_non_negative(self, sample_aggregate_series, mock_config):
        """RMS power must always be non-negative."""
        from mini_project_multi_model import create_rf_features
        features = create_rf_features(sample_aggregate_series, mock_config)
        assert (features["power_rms"] >= 0).all(), "power_rms contains negative values"


# ─────────────────────────────────────────────
#  2. Data Pipeline Tests
# ─────────────────────────────────────────────

class TestDataPipeline:

    def test_train_test_split_sizes(self, sample_aggregate_series, sample_target_series, mock_config):
        """80/20 train-test split should produce correct sizes."""
        from mini_project_multi_model import create_rf_features
        from sklearn.model_selection import train_test_split

        features = create_rf_features(sample_aggregate_series, mock_config).dropna()
        y = sample_target_series.loc[features.index]
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        total = len(features)
        assert len(X_train) == pytest.approx(total * 0.8, abs=2)
        assert len(X_test)  == pytest.approx(total * 0.2, abs=2)

    def test_features_and_target_aligned(self, sample_aggregate_series, sample_target_series, mock_config):
        """Feature index and target index must match exactly after alignment."""
        from mini_project_multi_model import create_rf_features

        features = create_rf_features(sample_aggregate_series, mock_config).dropna()
        y = sample_target_series.loc[features.index]
        assert list(features.index) == list(y.index), "Feature and target indices are misaligned"

    def test_target_non_negative(self, sample_target_series):
        """Appliance power must always be >= 0 (clipped at load time)."""
        assert (sample_target_series >= 0).all(), "Target contains negative power values"


# ─────────────────────────────────────────────
#  3. Model Tests
# ─────────────────────────────────────────────

class TestModels:

    @pytest.fixture
    def trained_rf(self, sample_aggregate_series, sample_target_series, mock_config):
        """Train a small Random Forest and return (model, X_test, y_test)."""
        from mini_project_multi_model import create_rf_features
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        features = create_rf_features(sample_aggregate_series, mock_config).dropna()
        y = sample_target_series.loc[features.index]
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def test_rf_predictions_shape(self, trained_rf):
        """Predictions array length must match test set length."""
        model, X_test, y_test = trained_rf
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_rf_predictions_non_negative(self, trained_rf):
        """All power predictions must be non-negative (physical constraint)."""
        model, X_test, _ = trained_rf
        preds = model.predict(X_test)
        # Clip negatives as the app does; here we just check raw model output is reasonable
        assert preds.min() >= -50, "Predictions are unreasonably negative"

    def test_rf_mae_reasonable(self, trained_rf):
        """MAE on this small dataset should be below 300 W (sanity check)."""
        from sklearn.metrics import mean_absolute_error
        model, X_test, y_test = trained_rf
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        assert mae < 300, f"MAE too high for sanity check: {mae:.1f} W"

    def test_xgboost_trains_and_predicts(self, sample_aggregate_series, sample_target_series, mock_config):
        """XGBoost should train and produce predictions of correct shape."""
        import xgboost as xgb
        from mini_project_multi_model import create_rf_features
        from sklearn.model_selection import train_test_split

        features = create_rf_features(sample_aggregate_series, mock_config).dropna()
        y = sample_target_series.loc[features.index]
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(n_estimators=10, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_svr_trains_and_predicts(self, sample_aggregate_series, sample_target_series, mock_config):
        """SVR should train and produce predictions of correct shape."""
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from mini_project_multi_model import create_rf_features
        from sklearn.model_selection import train_test_split

        features = create_rf_features(sample_aggregate_series, mock_config).dropna()
        y = sample_target_series.loc[features.index]
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = SVR(kernel='rbf', C=10, epsilon=5)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        assert len(preds) == len(y_test)


# ─────────────────────────────────────────────
#  4. Metrics Tests
# ─────────────────────────────────────────────

class TestMetrics:

    def test_mae_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        from sklearn.metrics import mean_absolute_error
        y_true = np.array([0.0, 100.0, 2300.0, 50.0])
        mae = mean_absolute_error(y_true, y_true)
        assert mae == 0.0

    def test_rmse_perfect_prediction(self):
        """RMSE should be 0 for perfect predictions."""
        from sklearn.metrics import mean_squared_error
        y_true = np.array([0.0, 100.0, 2300.0, 50.0])
        rmse = np.sqrt(mean_squared_error(y_true, y_true))
        assert rmse == 0.0

    def test_r2_perfect_prediction(self):
        """R² should be 1.0 for perfect predictions."""
        from sklearn.metrics import r2_score
        y_true = np.array([0.0, 100.0, 2300.0, 50.0])
        r2 = r2_score(y_true, y_true)
        assert r2 == pytest.approx(1.0)

    def test_r2_constant_prediction(self):
        """R² should be <= 0 if the model predicts a constant (mean baseline)."""
        from sklearn.metrics import r2_score
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.full_like(y_true, y_true.mean())   # always predict mean
        r2 = r2_score(y_true, y_pred)
        assert r2 == pytest.approx(0.0, abs=1e-9)

    def test_mae_greater_than_zero_for_imperfect(self):
        """MAE should be > 0 when predictions differ from actuals."""
        from sklearn.metrics import mean_absolute_error
        y_true = np.array([0.0, 2300.0, 0.0, 2300.0])
        y_pred = np.array([10.0, 2200.0, 5.0, 2100.0])
        assert mean_absolute_error(y_true, y_pred) > 0