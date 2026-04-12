from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
UNSUPERVISED_DIR = ROOT / "unsupervised"

UNSUPERVISED_DIR.mkdir(exist_ok=True)

def build_weighted_matrix(df):
    numerical = df[['distance_km', 'segment_count', 'transfer_count', 'is_peak_hour']].copy()
    numerical_scaled = StandardScaler().fit_transform(numerical)

    categorical = pd.get_dummies(
        df[['traffic_level', 'day_type', 'weather_condition', 'hour_block']],
        drop_first=False
    )
    categorical_scaled = StandardScaler().fit_transform(categorical) * 0.35

    return np.hstack([numerical_scaled, categorical_scaled])

def main():
    selection_df = pd.read_csv(RESULTS_DIR / 'unsupervised_k_selection.csv')
    labeled_df = pd.read_csv(RESULTS_DIR / 'transport_unsupervised_labeled.csv')

    profile_df = labeled_df.assign(
        traffic_num=labeled_df['traffic_level'].map({'low': 0, 'medium': 1, 'high': 2})
    ).groupby('cluster').agg(
        count=('trip_id', 'count'),
        avg_distance_km=('distance_km', 'mean'),
        avg_segments=('segment_count', 'mean'),
        avg_transfers=('transfer_count', 'mean'),
        peak_ratio=('is_peak_hour', 'mean'),
        avg_traffic_num=('traffic_num', 'mean'),
        mode_day=('day_type', lambda x: x.mode().iat[0]),
        mode_weather=('weather_condition', lambda x: x.mode().iat[0]),
        mode_hour=('hour_block', lambda x: x.mode().iat[0]),
        mode_traffic=('traffic_level', lambda x: x.mode().iat[0]),
    ).reset_index().sort_values('avg_distance_km')

    profile_df.to_csv(RESULTS_DIR / 'cluster_summary.csv', index=False)

    matrix = build_weighted_matrix(labeled_df)
    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(matrix)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(selection_df['k'], selection_df['inertia'], marker='o')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del codo para K-Means')

    plt.subplot(1, 2, 2)
    plt.scatter(projection[:, 0], projection[:, 1], c=labeled_df['cluster'], alpha=0.65)
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.title('Proyección 2D de los clusters')

    plt.tight_layout()
    plt.savefig(UNSUPERVISED_DIR / 'clusters_results.png', dpi=180, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(selection_df['k'], selection_df['silhouette'], marker='o')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score por valor de k')
    plt.tight_layout()
    plt.savefig(UNSUPERVISED_DIR / 'silhouette_scores.png', dpi=180, bbox_inches='tight')
    plt.close()

    print(profile_df.to_string(index=False))
    print('\nAnálisis de clusters completado correctamente.')

if __name__ == '__main__':
    main()