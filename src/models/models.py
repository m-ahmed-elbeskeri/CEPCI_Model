from prophet import Prophet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import joblib
import json
from abc import ABC, abstractmethod
import pickle
import optuna


@dataclass
class ValidationMetrics:
    """Stores validation metrics for model evaluation."""
    mape: float
    rmse: float
    r2: float
    window_type: str
    window_size: int
    forecast_horizon: int
    timestamp: datetime = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'mape': self.mape,
            'rmse': self.rmse,
            'r2': self.r2,
            'window_type': self.window_type,
            'window_size': self.window_size,
            'forecast_horizon': self.forecast_horizon,
            'timestamp': self.timestamp.isoformat()
        }


class BaseForecaster(ABC):
    """Abstract base class for forecasting models."""

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        pass


class CEPCIForecaster(BaseForecaster):
    """CEPCI forecasting model combining Prophet and Gaussian Process Regression."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the forecaster with configuration parameters.

        Args:
            config: Dictionary containing model parameters for Prophet and GPR
        """
        self.config = self._validate_config(config)
        self.prophet_model: Optional[Prophet] = None
        self.gpr_model: Optional[GaussianProcessRegressor] = None
        self.metrics_history: List[ValidationMetrics] = []
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default configuration parameters."""
        default_config = {
            'prophet': {
                'changepoint_prior_scale': 0.05,
                'yearly_seasonality': True,
                'seasonality_mode': 'multiplicative',
                'changepoint_range': 0.9,
                'n_changepoints': 25,
                'interval_width': 0.95
            },
            'gpr': {
                'n_restarts_optimizer': 15,
                'alpha': 0.1,
                'normalize_y': True
            },
            'validation': {
                'min_window_size': 20,
                'max_horizon': 5
            }
        }

        # Merge with provided config, keeping defaults for missing values
        for category in default_config:
            if category not in config:
                config[category] = {}
            config[category] = {**default_config[category], **config[category]}

        # Remove any invalid GPR parameters
        valid_gpr_params = {'n_restarts_optimizer', 'alpha', 'normalize_y'}
        config['gpr'] = {k: v for k, v in config['gpr'].items() if k in valid_gpr_params}

        return config

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare CEPCI data with advanced feature engineering for forecasting.

        Args:
            df: DataFrame with Year and CEPCI columns

        Returns:
            Prepared DataFrame with ds, y, and additional engineered features

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        required_cols = ['Year', 'CEPCI']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        if df.empty:
            raise ValueError("DataFrame is empty")

        if df[['Year', 'CEPCI']].isna().any().any():
            raise ValueError("Data contains missing values")

        try:
            # Sort and complete the series by filling any missing years
            df = df.sort_values('Year').reset_index(drop=True)
            df.set_index('Year', inplace=True)
            df = df.reindex(range(df.index.min(), df.index.max() + 1))
            df['CEPCI'] = df['CEPCI'].interpolate(method='linear')

            # Log transformation to stabilize variance
            df['CEPCI_log'] = np.log(df['CEPCI'])

            # Sine and Cosine transformations based on yearly data
            df['Year_sin'] = np.sin(2 * np.pi * (df.index - df.index.min()) / len(df))
            df['Year_cos'] = np.cos(2 * np.pi * (df.index - df.index.min()) / len(df))

            # Rolling statistics with larger windows
            df['CEPCI_rolling_mean_5'] = df['CEPCI'].rolling(window=5, min_periods=1).mean()
            df['CEPCI_rolling_std_5'] = df['CEPCI'].rolling(window=5, min_periods=1).std()
            df['CEPCI_rolling_median_5'] = df['CEPCI'].rolling(window=5, min_periods=1).median()

            # Lagged values to capture temporal dependencies
            df['CEPCI_lag_1'] = df['CEPCI'].shift(1)
            df['CEPCI_lag_2'] = df['CEPCI'].shift(2)

            # Difference transformations to remove non-stationarity
            df['CEPCI_diff_1'] = df['CEPCI'].diff(1)
            df['CEPCI_diff_2'] = df['CEPCI'].diff(2)

            # Polynomial terms to capture non-linear trends
            df['Year_squared'] = (df.index - df.index.min()) ** 2
            df['Year_cubed'] = (df.index - df.index.min()) ** 3

            # Reset index and rename for Prophet compatibility
            df.reset_index(inplace=True)
            df.rename(columns={'Year': 'ds', 'CEPCI': 'y'}, inplace=True)
            df['ds'] = pd.to_datetime(df['ds'].astype(str) + '-12-31')

            # Drop NaN values created by lagging and differencing
            prepared_df = df.dropna().reset_index(drop=True)

            return prepared_df[['ds', 'y', 'CEPCI_log', 'Year_sin', 'Year_cos',
                                'CEPCI_rolling_mean_5', 'CEPCI_rolling_std_5',
                                'CEPCI_rolling_median_5', 'CEPCI_lag_1', 'CEPCI_lag_2',
                                'CEPCI_diff_1', 'CEPCI_diff_2', 'Year_squared', 'Year_cubed']]
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def _optimize_prophet_hyperparameters(self, df: pd.DataFrame, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimise Prophet hyperparameters using Optuna.

        Args:
            df: DataFrame containing historical data for Prophet.
            n_trials: Number of trials for Optuna.

        Returns:
            Dictionary of best hyperparameters for Prophet.
        """

        def objective(trial):
            # Suggest hyperparameters
            changepoint_prior_scale = trial.suggest_loguniform('changepoint_prior_scale', 0.001, 0.5)
            seasonality_prior_scale = trial.suggest_loguniform('seasonality_prior_scale', 0.01, 10)
            seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])

            # Instantiate and fit Prophet model
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                seasonality_mode=seasonality_mode,
                yearly_seasonality=True
            )

            model.fit(df)

            # Predict and calculate MAPE
            forecast = model.predict(df)
            mape = mean_absolute_percentage_error(df['y'], forecast['yhat']) * 100

            return mape

        # Run Optuna optimisation
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        self.logger.info(f"Best Prophet hyperparameters: {best_params}")

        return best_params

    def _optimize_gpr_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimise GPR hyperparameters using Optuna.

        Args:
            X: Input features for GPR.
            y: Target residuals for GPR.
            n_trials: Number of trials for Optuna.

        Returns:
            Dictionary of best hyperparameters for GPR.
        """

        def objective(trial):
            # Suggest hyperparameters
            alpha = trial.suggest_loguniform('alpha', 0.001, 1.0)
            n_restarts_optimizer = trial.suggest_int('n_restarts_optimizer', 5, 20)

            # Set up kernel and GPR model
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                n_restarts_optimizer=n_restarts_optimizer,
                normalize_y=True
            )

            # Fit GPR model and evaluate with cross-validation
            model.fit(X, y)
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)

            return mse

        # Run Optuna optimisation
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        self.logger.info(f"Best GPR hyperparameters: {best_params}")

        return best_params

    def train(self, df: pd.DataFrame, prophet_trials: int = 50, gpr_trials: int = 50) -> None:
        """Train both Prophet and GPR models with Optuna-optimized parameters."""
        train_df = self.prepare_data(df)

        # Optimise and train Prophet
        prophet_params = self._optimize_prophet_hyperparameters(train_df, n_trials=prophet_trials)
        self.config['prophet'].update(prophet_params)

        self.prophet_model = Prophet(**self.config['prophet'])
        self.prophet_model.fit(train_df)

        # Calculate residuals for GPR
        historical_forecast = self.prophet_model.predict(train_df)
        residuals = train_df['y'].values - historical_forecast['yhat'].values

        # Optimise and train GPR
        X = np.arange(len(residuals)).reshape(-1, 1)
        gpr_params = self._optimize_gpr_hyperparameters(X, residuals, n_trials=gpr_trials)
        self.config['gpr'].update(gpr_params)

        # Initialize kernel and create GPR model with best parameters
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
        self.gpr_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.config['gpr']['n_restarts_optimizer'],
            alpha=self.config['gpr']['alpha'],
            normalize_y=self.config['gpr']['normalize_y']
        )
        self.gpr_model.fit(X, residuals)

    def predict(self, df: pd.DataFrame, forecast_horizon: int = 36) -> pd.DataFrame:
        """Generate forecasts with uncertainty estimates, ensuring CEPCI remains non-negative."""
        if self.prophet_model is None or self.gpr_model is None:
            raise ValueError("Models must be trained before prediction")

        # Generate future dates
        last_date = pd.to_datetime(df['Year'].max(), format='%Y')
        future_dates = pd.date_range(
            start=last_date,
            periods=forecast_horizon + 1,
            freq='Y'
        )

        # Prophet forecast
        future_df = pd.DataFrame({'ds': future_dates})
        prophet_forecast = self.prophet_model.predict(future_df)

        # GPR forecast
        X_future = np.arange(
            len(df),
            len(df) + len(future_dates)
        ).reshape(-1, 1)
        gpr_predictions, gpr_std = self.gpr_model.predict(X_future, return_std=True)

        # Combine forecasts and ensure non-negativity
        forecast_df = pd.DataFrame({
            'Year': future_dates.year,
            'CEPCI_Predicted': np.maximum(0, prophet_forecast['yhat'] + gpr_predictions),
            'CEPCI_Lower': np.maximum(0, prophet_forecast['yhat_lower'] + (gpr_predictions - 2 * gpr_std)),
            'CEPCI_Upper': np.maximum(0, prophet_forecast['yhat_upper'] + (gpr_predictions + 2 * gpr_std)),
            'Prophet_Forecast': np.maximum(0, prophet_forecast['yhat']),
            'GPR_Adjustment': gpr_predictions
        })

        return forecast_df

    def save(self, path: Path) -> None:
        """Save model state and configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Prophet model using pickle
        with open(path / 'prophet_model.pkl', 'wb') as f:
            pickle.dump(self.prophet_model, f)

        # Save GPR model
        joblib.dump(self.gpr_model, path / 'gpr_model.joblib')

        # Save configuration and metrics
        with open(path / 'config.json', 'w') as f:
            json.dump(self.config, f)

        # Save metrics history as JSON
        with open(path / 'metrics_history.json', 'w') as f:
            json.dump([m.to_dict() for m in self.metrics_history], f)
    def load(self, path: Path) -> None:
        """Load model state and configuration."""
        path = Path(path)

        # Load Prophet model
        with open(path / 'prophet_model.json', 'r') as f:
            self.prophet_model = Prophet.from_json(json.load(f))

        # Load GPR model
        self.gpr_model = joblib.load(path / 'gpr_model.joblib')

        # Load configuration and metrics
        with open(path / 'config.json', 'r') as f:
            self.config = json.load(f)

        with open(path / 'metrics_history.json', 'r') as f:
            metrics_data = json.load(f)
            self.metrics_history = [
                ValidationMetrics(**m) for m in metrics_data
            ]
