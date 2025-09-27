import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv(r"C:\Users\guilherme.orlandi\Documents\GitHub\Exercicio01-Machine-Learning\docs\data\fifa_ranking.csv")

print("Primeiras linhas do dataset:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

plt.hist(df["rank"], bins=30, color="skyblue", edgecolor="black")
plt.title("Distribuição do Ranking das Seleções")
plt.xlabel("Rank")
plt.ylabel("Frequência")
plt.savefig("distribuicao_rank.png")
plt.close()


df = df.dropna()

X = df[["previous_points", "rank_change", "confederation"]].copy()
X = pd.get_dummies(X, drop_first=True)

y = pd.cut(df["rank"], bins=[0,50,100,150,200,300], labels=[1,2,3,4,5]) 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

print("\n--- Avaliação KNN ---")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred, average="macro"))
print("Recall:", recall_score(y_test, y_pred, average="macro"))
print("F1-Score:", f1_score(y_test, y_pred, average="macro"))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

y_bin = (y_test == 1).astype(int)
y_pred_proba = knn.predict_proba(X_test)[:, 0]
roc_auc = roc_auc_score(y_bin, y_pred_proba)
print("ROC-AUC (classe 1 vs resto):", roc_auc)

fpr, tpr, _ = roc_curve(y_bin, y_pred_proba)
plt.plot(fpr, tpr, color="blue")
plt.plot([0,1],[0,1], color="red", linestyle="--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC - KNN (classe 1 vs resto)")
plt.savefig("roc_knn.png")
plt.close()


k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

df["cluster"] = kmeans.labels_

print("\n--- Avaliação K-Means ---")
print("Número de pontos em cada cluster:")
print(df["cluster"].value_counts())
print("\nMédia das variáveis por cluster:")
print(df.groupby("cluster")[["previous_points", "rank_change"]].mean())

sil_score = silhouette_score(X_scaled, kmeans.labels_)
print("\nSilhouette Score:", sil_score)

jitter_strength = 0.5
X_jittered = X_scaled + np.random.uniform(-jitter_strength, jitter_strength, X_scaled.shape)

plt.scatter(X_jittered[:, 0], X_jittered[:, 1], c=df["cluster"], cmap="viridis", s=30, edgecolor="k")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", s=200, marker="X")
plt.xlabel("Previous Points")
plt.ylabel("Rank Change")
plt.title(f"K-Means com k={k} (com jitter)")
plt.savefig("kmeans_clusters.png")
plt.close()
