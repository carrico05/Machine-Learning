import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# exploração dos dados
df = pd.read_csv(r"C:\Users\guilherme.orlandi\Documents\GitHub\Exercicio01-Machine-Learning\docs\data\fifa_ranking.csv")

print("===== Primeiras linhas do dataset =====")
print(df.head())

print("\n===== Informações gerais =====")
print(df.info())

print("\n===== Estatísticas descritivas =====")
print(df.describe())

plt.figure(figsize=(8,5))
plt.hist(df["rank"], bins=30, color="skyblue", edgecolor="black")
plt.title("Distribuição do Rank das Seleções")
plt.xlabel("Rank")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("rf_distribuicao_rank.png")
plt.close()

# pré-processamento 
df = df.dropna()

X = df[["previous_points", "rank_change", "confederation"]].copy()
X = pd.get_dummies(X, drop_first=True)
y = pd.cut(df["rank"], bins=[0,50,100,150,200,300], labels=[1,2,3,4,5])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# treinamento do modelo
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    oob_score=True,   
    random_state=42
)
rf.fit(X_train, y_train)

# avaliação do modelo
y_pred = rf.predict(X_test)

print("\n===== Avaliação do Modelo Random Forest =====")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("OOB Score:", rf.oob_score_)
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.savefig("rf_matriz_confusao.png")
plt.close()

# importância das variáveis
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,5))
importances.plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title("Importância das Variáveis")
plt.xlabel("Variáveis")
plt.ylabel("Importância")
plt.tight_layout()
plt.savefig("rf_importancia_features.png")
plt.close()

print("\n===== Importância das Variáveis =====")
print(importances)


# acurácia
n_estimators_range = range(1, 101)
train_acc = []
test_acc = []

for n in n_estimators_range:
    rf_temp = RandomForestClassifier(
        n_estimators=n,
        max_depth=5,
        random_state=42
    )
    rf_temp.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, rf_temp.predict(X_train)))
    test_acc.append(accuracy_score(y_test, rf_temp.predict(X_test)))

plt.figure(figsize=(8,5))
plt.plot(n_estimators_range, train_acc, label="Treino", color="blue")
plt.plot(n_estimators_range, test_acc, label="Teste", color="red")
plt.xlabel("Número de Árvores")
plt.ylabel("Acurácia")
plt.title("Acurácia vs Número de Árvores")
plt.legend()
plt.tight_layout()
plt.savefig("rf_acuracia_vs_arvores.png")
plt.close()
