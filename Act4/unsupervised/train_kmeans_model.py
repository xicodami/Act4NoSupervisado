from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transport_unsupervised_dataset.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = [
    'distance_km', 'segment_count', 'transfer_count', 'is_peak_hour',
    'traffic_level', 'day_type', 'weather_condition', 'hour_block'
]

def build_weighted_matrix(df):
    numerical = df[['distance_km', 'segment_count', 'transfer_count', 'is_peak_hour']].copy()
    numerical_scaled = StandardScaler().fit_transform(numerical)

    categorical = pd.get_dummies(
        df[['traffic_level', 'day_type', 'weather_condition', 'hour_block']],
        drop_first=False
    )
    categorical_scaled = StandardScaler().fit_transform(categorical) * 0.35

    matrix = np.hstack([numerical_scaled, categorical_scaled])

    metadata = {
        'numerical_columns': list(numerical.columns),
        'categorical_dummy_columns': list(categorical.columns),
        'categorical_weight': 0.35
    }
    return matrix, metadata

def main():
    df = pd.read_csv(DATA_PATH)

    matrix, metadata = build_weighted_matrix(df)

    search_rows = []
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)
        silhouette = silhouette_score(matrix, labels, sample_size=min(700, len(df)), random_state=42)

        search_rows.append({
            'k': k,
            'inertia': kmeans.inertia_,
            'silhouette': silhouette
        })

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(RESULTS_DIR / 'unsupervised_k_selection.csv', index=False)

    chosen_k = 5
    final_model = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    labels = final_model.fit_predict(matrix)

    labeled_df = df.copy()
    labeled_df['cluster'] = labels
    labeled_df.to_csv(RESULTS_DIR / 'transport_unsupervised_labeled.csv', index=False)

    model_bundle = {
        'model': final_model,
        'matrix_metadata': metadata
    }
    joblib.dump(model_bundle, MODELS_DIR / 'kmeans_transport.joblib')

    print(search_df.to_string(index=False))
    print(f"\nNúmero de clusters seleccionado: {chosen_k}")

if __name__ == '__main__':
    main()