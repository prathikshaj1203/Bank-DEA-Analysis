import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def cluster_banks(df: pd.DataFrame, features: list[str], n_clusters: int = 3):
    """Cluster banks and return dataframe with cluster labels + PCA coords."""
    X = df[features].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    result = df.copy()
    result["cluster"] = labels
    result["pca1"] = coords[:, 0]
    result["pca2"] = coords[:, 1]

    return result, kmeans
