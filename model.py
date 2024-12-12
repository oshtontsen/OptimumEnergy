import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for temporal features to prevent data leakage"""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Create cyclical features
        for col, period in [
            ("hour", 24),
            ("day_of_week", 7),
            ("month", 12),
            ("day_of_year", 365),
        ]:
            if col in X_copy.columns:
                X_copy[f"{col}_sin"] = np.sin(2 * np.pi * X_copy[col] / period)
                X_copy[f"{col}_cos"] = np.cos(2 * np.pi * X_copy[col] / period)

        # Drop original temporal columns
        X_copy = X_copy.drop(
            ["hour", "day_of_week", "month", "day_of_year"], axis=1, errors="ignore"
        )

        return X_copy


def create_lagged_features(df, columns, lags):
    """
    Create lagged features using only past data
    """
    df_copy = df.copy()

    for col in columns:
        for lag in lags:
            df_copy[f"{col}_lag_{lag}"] = df_copy[col].shift(lag)

    return df_copy


def preprocess_data(dataframe):
    """
    Enhanced preprocessing with feature engineering and data leakage prevention
    """
    # Sort by timestamp first to ensure correct temporal ordering
    dataframe = dataframe.sort_values("timestamp")

    # Convert timestamp to datetime
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])

    # Extract basic time features
    dataframe["hour"] = dataframe["timestamp"].dt.hour
    dataframe["day_of_week"] = dataframe["timestamp"].dt.dayofweek
    dataframe["month"] = dataframe["timestamp"].dt.month
    dataframe["day_of_year"] = dataframe["timestamp"].dt.dayofyear

    # Create lagged features (using only past data)
    lag_features = ["temperature", "humidity"]
    lags = [1, 2, 3, 6]  # Using only past hours
    dataframe = create_lagged_features(dataframe, lag_features, lags)

    # Create interaction features
    dataframe["temp_humidity"] = dataframe["temperature"] * dataframe["humidity"]

    # Add polynomial features for temperature and humidity
    dataframe["temperature_squared"] = dataframe["temperature"] ** 2
    dataframe["humidity_squared"] = dataframe["humidity"] ** 2

    # Create time windows for peak/off-peak hours
    dataframe["is_peak_hours"] = (
        (dataframe["hour"] >= 9) & (dataframe["hour"] <= 17)
    ).astype(int)

    # Drop rows with NaN values created by lagged features
    dataframe = dataframe.dropna()

    print("Missing Values:")
    print(dataframe.isnull().sum())

    return dataframe


def prepare_features(df):
    """
    Prepare features ensuring no data leakage
    """
    features = [
        "temperature",
        "humidity",
        "hour",
        "day_of_week",
        "day_of_year",
        "temp_humidity",
        "temperature_squared",
        "humidity_squared",
        "is_peak_hours",
    ]

    # Add lagged feature names
    lag_features = ["temperature", "humidity"]
    lags = [1, 2, 3, 6]
    for col in lag_features:
        for lag in lags:
            features.append(f"{col}_lag_{lag}")

    return features


def create_model_pipelines():
    """
    Create pipelines for all models with proper feature transformation
    """
    models = {
        "Linear Regression": Pipeline(
            [
                ("temporal_features", TemporalFeatureTransformer()),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("temporal_features", TemporalFeatureTransformer()),
                ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("temporal_features", TemporalFeatureTransformer()),
                ("model", GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ]
        ),
        "SVR": Pipeline(
            [
                ("temporal_features", TemporalFeatureTransformer()),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("temporal_features", TemporalFeatureTransformer()),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return models


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate all models using pipelines
    """
    models = create_model_pipelines()
    results = {}

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results[name] = {
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "MAPE": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        }

    return results


def plot_correlation_heatmap(df_processed):
    """Plot correlation heatmap of features"""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_processed[
        [
            "temperature",
            "humidity",
            "energy_consumption",
            "hour",
            "day_of_week",
            "month",
        ]
    ].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_scatter_relationships(df_processed):
    """Plot scatter plots for key relationships"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Temperature vs Energy Consumption
    axs[0, 0].scatter(
        df_processed["temperature"], df_processed["energy_consumption"], alpha=0.5
    )
    axs[0, 0].set_title("Temperature vs Energy Consumption")
    axs[0, 0].set_xlabel("Temperature")
    axs[0, 0].set_ylabel("Energy Consumption")

    # Humidity vs Energy Consumption
    axs[0, 1].scatter(
        df_processed["humidity"], df_processed["energy_consumption"], alpha=0.5
    )
    axs[0, 1].set_title("Humidity vs Energy Consumption")
    axs[0, 1].set_xlabel("Humidity")
    axs[0, 1].set_ylabel("Energy Consumption")

    # Hour vs Energy Consumption
    axs[1, 0].scatter(
        df_processed["hour"], df_processed["energy_consumption"], alpha=0.5
    )
    axs[1, 0].set_title("Hour vs Energy Consumption")
    axs[1, 0].set_xlabel("Hour of Day")
    axs[1, 0].set_ylabel("Energy Consumption")

    # Additional scatter plot: Temperature_squared vs Energy Consumption
    axs[1, 1].scatter(
        df_processed["temperature_squared"],
        df_processed["energy_consumption"],
        alpha=0.5,
    )
    axs[1, 1].set_title("Temperature² vs Energy Consumption")
    axs[1, 1].set_xlabel("Temperature²")
    axs[1, 1].set_ylabel("Energy Consumption")

    plt.tight_layout()
    plt.show()


def plot_model_comparison(results):
    """
    Plot comparison of all models' performances
    """
    metrics = ["RMSE", "MAE", "R2", "MAPE"]
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 6))

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in results.keys()]
        axes[i].bar(results.keys(), values)
        axes[i].set_title(f"Model Comparison - {metric}")
        axes[i].set_xticklabels(results.keys(), rotation=45)

    plt.tight_layout()
    plt.show()


def main():
    # Load the dataset
    df = pd.read_csv("energy_consumption_data.csv")

    # Preprocess data
    df_processed = preprocess_data(df)

    # Get features
    features = prepare_features(df_processed)

    # Create data visualizations
    plot_correlation_heatmap(df_processed)
    plot_scatter_relationships(df_processed)

    X = df_processed[features]
    y = df_processed["energy_consumption"]

    # Split data chronologically
    train_size = int(len(df_processed) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Print results
    print("\nModel Performance:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Plot model comparison
    plot_model_comparison(results)


if __name__ == "__main__":
    main()
