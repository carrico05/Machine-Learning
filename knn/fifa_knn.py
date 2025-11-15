import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Carregar os dados
df = pd.read_csv(r"C:\Users\guilherme.orlandi\Documents\GitHub\Exercicio01-Machine-Learning\docs\data\fifa_ranking.csv") 

print("Primeiras linhas do dataset:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

# Visualização simples: distribuição dos ranks
#plt.figure(figsize=(8,5))
#plt.hist(df["rank"], bins=30, color="lightgreen", edgecolor="black")
#plt.title("Distribuição do Rank das Seleções")
#plt.xlabel("Rank")
#plt.ylabel("Frequência")
#plt.savefig("knn_distribuicao_rank.png")
#plt.close()

# Pré-processamento
df = df.dropna()

# Features escolhidas
X = df[["previous_points", "rank_change", "confederation"]].copy()

# Converter confederação em variável numérica
X = pd.get_dummies(X, drop_first=True)

# Variável alvo: faixas de ranking
y = pd.cut(df["rank"], bins=[0,50,100,150,200,300], labels=[1,2,3,4,5])

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinamento do Modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Avaliação do Modelo
y_pred = knn.predict(X_test)

print("\nAcurácia:", accuracy_score(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))


#cm = confusion_matrix(y_test, y_pred)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
#disp.plot(cmap="Greens")
#plt.title("Matriz de Confusão - KNN")
# plt.savefig("knn_matriz_confusao.png")
#plt.close()

# Selecionando apenas duas features
X_vis = df[["previous_points", "rank_change"]].copy()



# Reajustando a variável alvo após remover outliers
y_vis = y.loc[X_vis.index]

# Normalizando
scaler = StandardScaler()
X_vis_scaled = scaler.fit_transform(X_vis)

# Train/test split
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis_scaled, y_vis, test_size=0.2, random_state=42
)

# Modelo KNN
knn_vis = KNeighborsClassifier(n_neighbors=5)
knn_vis.fit(X_train_vis, y_train_vis)

# Grade de pontos
h = 0.05
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array(Z).reshape(xx.shape)

# Adicionar jitter para melhor vizualização
jitter_strength = 1.0
X_vis_jittered = X_vis_scaled + np.random.uniform(
    -jitter_strength, jitter_strength, X_vis_scaled.shape
)

# Plot decision boundary
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
plt.scatter(
    X_vis_jittered[:, 0], X_vis_jittered[:, 1],
    c=y_vis, edgecolor='k', cmap=plt.cm.Set1,
    alpha=0.7, s=20)  
plt.xlabel("Previous Points (normalizado)")
plt.ylabel("Rank Change (normalizado)")
plt.title("Decision Boundary - KNN (sem outliers)")
plt.savefig("knn_decision_boundary.png")
plt.close()