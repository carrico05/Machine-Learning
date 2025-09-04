import pandas as pd
import matplotlib.pyplot as plt
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
plt.figure(figsize=(8,5))
plt.hist(df["rank"], bins=30, color="lightgreen", edgecolor="black")
plt.title("Distribuição do Rank das Seleções")
plt.xlabel("Rank")
plt.ylabel("Frequência")
plt.savefig("knn_distribuicao_rank.png")
plt.close()

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

# 3. Divisão treino/teste
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


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap="Greens")
plt.title("Matriz de Confusão - KNN")
plt.savefig("knn_matriz_confusao.png")
plt.close()
