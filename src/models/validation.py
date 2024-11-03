# src/validation.py
from sklearn.metrics import mean_squared_error
import numpy as np
import logging
from .models import train_prophet, train_gpr


def validate_predictions(predictions, name):
    """Ensure predictions contain no NaN values."""
    if np.isnan(predictions).any():
        logging.warning(f"{name} predictions contain NaNs. Filling NaNs with zero.")
        return np.nan_to_num(predictions)
    return predictions


def sliding_window_validation(data, config, prophet_config, gpr_config):
    """Perform sliding window validation for hybrid Prophet-GPR model."""
    prophet_errors = []  # Initialize separate lists for Prophet and Hybrid errors
    hybrid_errors = []
    horizon = config['forecast_horizon']
    window_size = config['window_size']

    for i in range(window_size, len(data) - horizon, horizon):
        train, test = data.iloc[i - window_size:i], data.iloc[i:i + horizon]

        # Train Prophet model
        train_df = train.reset_index().rename(columns={'Date': 'ds', 'CEPCI': 'y'})
        prophet_forecast, _ = train_prophet(train_df, prophet_config)

        # Calculate residuals
        residuals = train['CEPCI'] - prophet_forecast['yhat'][:len(train_df)]
        residuals = residuals.fillna(0)  # Ensure no NaNs in residuals

        # Train GPR on residuals
        gpr_model = train_gpr(residuals, gpr_config)
        X_test = np.arange(len(residuals), len(residuals) + horizon).reshape(-1, 1)
        gpr_pred = gpr_model.predict(X_test)

        # Check predictions for NaNs before calculating MSE
        prophet_pred = validate_predictions(prophet_forecast['yhat'][-horizon:], "Prophet")
        hybrid_pred = validate_predictions(prophet_pred + gpr_pred, "Hybrid")

        prophet_errors.append(mean_squared_error(test['CEPCI'], prophet_pred))
        hybrid_errors.append(mean_squared_error(test['CEPCI'], hybrid_pred))
        logging.info(f"Sliding Window {i // horizon}: Prophet MSE={prophet_errors[-1]}, Hybrid MSE={hybrid_errors[-1]}")

    return np.mean(prophet_errors), np.mean(hybrid_errors)


def expanding_window_validation(data, config, prophet_config, gpr_config):
    """Perform expanding window validation for hybrid Prophet-GPR model."""
    prophet_errors = []  # Initialize separate lists for Prophet and Hybrid errors
    hybrid_errors = []
    horizon = config['forecast_horizon']
    initial_window = config['initial_window']

    for i in range(initial_window, len(data) - horizon, horizon):
        train, test = data.iloc[:i], data.iloc[i:i + horizon]

        # Train Prophet model
        train_df = train.reset_index().rename(columns={'Date': 'ds', 'CEPCI': 'y'})
        prophet_forecast, _ = train_prophet(train_df, prophet_config)

        # Calculate residuals
        residuals = train['CEPCI'] - prophet_forecast['yhat'][:len(train_df)]
        residuals = residuals.fillna(0)  # Ensure no NaNs in residuals

        # Train GPR on residuals
        gpr_model = train_gpr(residuals, gpr_config)
        X_test = np.arange(len(residuals), len(residuals) + horizon).reshape(-1, 1)
        gpr_pred = gpr_model.predict(X_test)

        # Check predictions for NaNs before calculating MSE
        prophet_pred = validate_predictions(prophet_forecast['yhat'][-horizon:], "Prophet")
        hybrid_pred = validate_predictions(prophet_pred + gpr_pred, "Hybrid")

        prophet_errors.append(mean_squared_error(test['CEPCI'], prophet_pred))
        hybrid_errors.append(mean_squared_error(test['CEPCI'], hybrid_pred))
        logging.info(
            f"Expanding Window {i // horizon}: Prophet MSE={prophet_errors[-1]}, Hybrid MSE={hybrid_errors[-1]}")

    return np.mean(prophet_errors), np.mean(hybrid_errors)
