import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar os dados
df = pd.read_csv("fifa_ranking_2024.csv")

# Exploração inicial
print("Primeiras linhas do dataset:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

# distribuição do rank
plt.figure(figsize=(8,5))
plt.hist(df["rank"], bins=30, color="skyblue", edgecolor="black")
plt.title("Distribuição do Rank das Seleções")
plt.xlabel("Rank")
plt.ylabel("Frequência")
plt.show()

# evolução do ranking de algumas seleções
teams_to_plot = ["Brazil", "Germany", "Argentina"]
plt.figure(figsize=(10,6))
for team in teams_to_plot:
    team_data = df[df["team"] == team]
    plt.plot(pd.to_datetime(team_data["rank_date"]), team_data["rank"], label=team)
plt.gca().invert_yaxis() 
plt.title("Evolução do Ranking FIFA")
plt.xlabel("Ano")
plt.ylabel("Posição no Ranking")
plt.legend()
plt.show()

# Pré-processamento
df = df.dropna()

# Features selecionadas
X = df[["previous_points", "rank_change", "confederation"]].copy()

# Transformar confederação em numérica
X = pd.get_dummies(X, drop_first=True)

# Target: faixas de ranking
y = pd.cut(df["rank"], bins=[0,50,100,150,200,300],
           labels=[1,2,3,4,5])

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)

print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
