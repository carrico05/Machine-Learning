import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Carregar os dados
df = pd.read_csv(r"C:\Users\guilherme.orlandi\Documents\GitHub\Exercicio01-Machine-Learning\docs\data\fifa_ranking.csv")
df = df.dropna()

# Selecionar features
X = df[["previous_points", "rank_change"]]

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método do cotovelo
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inércia")
plt.title("Método do Cotovelo")
plt.savefig("kmeans_cotovelo.png")
plt.close()

# Treinamento do modelo (KMeans com k=5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Adicionar rótulos dos clusters
df["cluster"] = kmeans.labels_

# Visualizar clusters com jitter
jitter_strength = 0.5
X_jittered = X_scaled + np.random.uniform(-jitter_strength, jitter_strength, X_scaled.shape)

plt.scatter(X_jittered[:, 0], X_jittered[:, 1], c=df["cluster"], cmap="viridis", s=30, edgecolor="k")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", s=200, marker="X")
plt.xlabel("Previous Points")
plt.ylabel("Rank Change")
plt.title(f"K-Means com k={k} (com jitter)")
plt.savefig("kmeans_clusters.png")
plt.close()

# Análises
print("Número de pontos em cada cluster:")
print(df["cluster"].value_counts())

print("\nMédia das variáveis em cada cluster:")
print(df.groupby("cluster")[["previous_points", "rank_change"]].mean())
