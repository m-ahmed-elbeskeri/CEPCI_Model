# main.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import sys
from datetime import datetime
import json
from typing import Optional, Dict, Any
from src.models.models import CEPCIForecaster, ValidationMetrics
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import numpy as np

def setup_logging(log_path: Optional[Path] = None) -> None:
    """Configure logging with both file and console handlers."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(log_path / f'cepci_forecast_{datetime.now():%Y%m%d_%H%M%S}.log'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        raise


def validate_data(df: pd.DataFrame) -> None:
    """Validate input data quality."""
    if df.empty:
        raise ValueError("Empty dataset")

    required_cols = ['Year', 'CEPCI']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df[required_cols].isna().any().any():
        raise ValueError("Dataset contains missing values")

    if not df['Year'].is_monotonic_increasing:
        raise ValueError("Years must be in chronological order")

    # Check for duplicates
    if df['Year'].duplicated().any():
        raise ValueError("Duplicate years found in dataset")

    # Check for outliers using IQR method
    Q1 = df['CEPCI'].quantile(0.25)
    Q3 = df['CEPCI'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[
        (df['CEPCI'] < (Q1 - 1.5 * IQR)) |
        (df['CEPCI'] > (Q3 + 1.5 * IQR))
        ]

    if not outliers.empty:
        logging.warning(f"Potential outliers detected in years: {outliers['Year'].tolist()}")


def plot_forecast(
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        metrics: Dict[str, float],
        output_path: Path
) -> None:
    """Create and save forecast visualization."""
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot historical data
    sns.scatterplot(
        data=historical_df,
        x='Year',
        y='CEPCI',
        color='black',
        label='Historical Data',
        ax=ax
    )

    # Plot forecast
    ax.plot(
        forecast_df['Year'],
        forecast_df['CEPCI_Predicted'],
        'b-',
        label='Hybrid Forecast',
        linewidth=2
    )

    # Add confidence intervals
    ax.fill_between(
        forecast_df['Year'],
        forecast_df['CEPCI_Lower'],
        forecast_df['CEPCI_Upper'],
        alpha=0.2,
        color='blue',
        label='95% Confidence Interval'
    )

    # Add Prophet and GPR components
    ax.plot(
        forecast_df['Year'],
        forecast_df['Prophet_Forecast'],
        'g--',
        label='Prophet Component',
        alpha=0.5
    )

    # Customize plot
    ax.set_title('CEPCI Forecast with Uncertainty Bounds', fontsize=14, pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('CEPCI Value', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add metrics annotation
    metrics_text = (
        f"Validation Metrics:\n"
        f"MAPE: {metrics['mape']:.2f}%\n"
        f"RMSE: {metrics['rmse']:.2f}\n"
        f"R²: {metrics['r2']:.3f}"
    )
    plt.text(
        0.02, 0.98,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path / 'forecast_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def calculate_validation_metrics(
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        window_size: int,
        forecast_horizon: int
) -> ValidationMetrics:
    """Calculate validation metrics for the forecast."""
    # Get overlapping period
    overlap_df = pd.merge(
        historical_df,
        forecast_df,
        on='Year',
        how='inner'
    )

    if len(overlap_df) == 0:
        raise ValueError("No overlapping periods between historical and forecast data")

    actual = overlap_df['CEPCI']
    predicted = overlap_df['CEPCI_Predicted']

    return ValidationMetrics(
        mape=mean_absolute_percentage_error(actual, predicted) * 100,
        rmse=np.sqrt(mean_squared_error(actual, predicted)),
        r2=r2_score(actual, predicted),
        window_type='rolling',
        window_size=window_size,
        forecast_horizon=forecast_horizon
    )


def main() -> None:
    """Main execution function."""
    # Setup paths
    base_path = Path(__file__).parent
    config_path = base_path / 'config' / 'config.json'
    data_path = base_path / 'data' / 'raw' /'historical_cepci.xlsx'
    output_path = base_path / 'output'
    log_path = base_path / 'logs'

    # Create directories
    for path in [output_path, log_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(config_path)

        # Load and validate data
        logger.info("Loading historical data...")
        historical_df = pd.read_excel(data_path)
        validate_data(historical_df)

        # Initialize and train model
        logger.info("Initializing forecaster...")
        forecaster = CEPCIForecaster(config)

        logger.info("Training model...")
        forecaster.train(historical_df)

        # Generate forecast
        logger.info("Generating forecast...")
        forecast_horizon = config.get('forecast_horizon', 36)  # 3 years by default
        forecast_df = forecaster.predict(historical_df, forecast_horizon)

        # Calculate metrics
        logger.info("Calculating validation metrics...")
        metrics = calculate_validation_metrics(
            historical_df,
            forecast_df,
            window_size=config['validation']['min_window_size'],
            forecast_horizon=forecast_horizon
        )

        # Save results
        logger.info("Saving results...")
        forecast_df.to_csv(output_path / 'forecast_results.csv', index=False)

        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics.to_dict(), f, indent=4)

        # Create visualization
        logger.info("Creating visualization...")
        plot_forecast(
            historical_df,
            forecast_df,
            metrics.to_dict(),
            output_path
        )

        # Save model
        logger.info("Saving model...")
        model_path = output_path / 'model'
        forecaster.save(model_path)

        logger.info("Forecast completed successfully!")

    except Exception as e:
        logger.error(f"Error during forecast generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()