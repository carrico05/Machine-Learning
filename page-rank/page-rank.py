import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

df = pd.read_csv(r"C:\Users\guilherme.orlandi\Documents\GitHub\Exercicio01-Machine-Learning\docs\data\fifa_ranking.csv")

print("===== Primeiras linhas do dataset =====")
print(df.head())

print("\n===== Informações gerais =====")
print(df.info())

print("\n===== Estatísticas descritivas =====")
print(df.describe())

df = df.dropna().reset_index(drop=True)

plt.figure(figsize=(8,5))
plt.hist(df["rank"], bins=30, edgecolor="black")
plt.title("Distribuição do Rank das Seleções")
plt.xlabel("Rank")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("pr_distribuicao_rank.png")
plt.close()

print("\n===== Construção do Grafo =====")

import networkx as nx
G = nx.DiGraph()

# ordenar por rank
df = df.sort_values("rank").reset_index(drop=True)

# adicionar nós
for _, row in df.iterrows():
    G.add_node(
        row["country_full"],
        rank=row["rank"],
        conf=row["confederation"],
        points=row["total_points"]
    )

# criar arestas sem usar O(n²)
# um país aponta para o país melhor colocado na mesma confederação
groups = df.groupby("confederation")

for conf, group in groups:
    group = group.sort_values("rank")
    countries = list(group["country_full"].values)

    # ligar cada país ao país imediatamente melhor rankeado
    for i in range(1, len(countries)):
        G.add_edge(countries[i], countries[i-1])

print("Número de nós:", G.number_of_nodes())
print("Número de arestas:", G.number_of_edges())


# Função para imprimir o grafo sem travar o terminal

def print_graph(G):
    visited = set()
    for node in G.nodes:
        if node not in visited:
            _print_node(G, node, visited)

def _print_node(G, node, visited, depth=0):
    indent = "  " * depth
    print(f"{indent}{node}")
    visited.add(node)

    for neighbor in G.successors(node):
        if neighbor not in visited:
            _print_node(G, neighbor, visited, depth + 1)


print("\n===== Estrutura do Grafo (visualização segura) =====")
print_graph(G)

# Remoção dos dead ends isolados
G = nx.DiGraph(G)

# ================
# PageRank manual
# ================

def pagerank_manual(G, d=0.85, tol=0.0001, max_iter=100):
    nodes = list(G.nodes())
    n = len(nodes)
    pr = {node: 1/n for node in nodes}
    
    for _ in range(max_iter):
        pr_new = {}
        for node in nodes:
            incoming = G.in_edges(node)
            s = 0
            for a, b in incoming:
                out_degree = G.out_degree(a)
                if out_degree > 0:
                    s += pr[a] / out_degree
            pr_new[node] = (1 - d)/n + d * s
            
        diff = sum(abs(pr_new[node] - pr[node]) for node in nodes)
        pr = pr_new
        if diff < tol:
            break
        
    return pr


print("\n===== Cálculando PageRank Manual =====")
pr_manual = pagerank_manual(G, d=0.85)


pr_manual_sorted = sorted(pr_manual.items(), key=lambda x: x[1], reverse=True)


print("\n===== Top 10 Seleções por PageRank Manual =====")
for i, (team, score) in enumerate(pr_manual_sorted[:10], 1):
    print(f"{i}. {team} - {score:.6f}")
    

# Comparação com networkx
print("\n===== PageRank do NetworkX =====")
pr_nx = nx.pagerank(G, alpha=0.85)

pr_nx_sorted = sorted(pr_nx.items(), key=lambda x: x[1], reverse=True)


print("\n===== Top 10 Seleções por Networkx =====")
for i, (team, score) in enumerate(pr_nx_sorted[:10], 1):
    print(f"{i}. {team} - {score:.6f}")
    
    
# Comparação dos resultados
top10_manual = dict(pr_manual_sorted[:10])


plt.figure(figsize=(10,5))
plt.bar(top10_manual.keys(), top10_manual.values())
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 PageRank Manual")
plt.tight_layout()
plt.savefig("pr_top10_manual.png")
plt.close()


# Impacto do damping factor
print("\n===== Impacto do Damping Factor =====")
d_values = [0.5, 0.85, 0.99]
scores_by_d = {}

for d in d_values:
    pr_d = pagerank_manual(G, d=d)
    ordered = sorted(pr_d.items(), key=lambda x: x[1], reverse=True)
    scores_by_d[d] = ordered[:10]
    
    print(f"\n--- d = {d} ---")
    for i, (team, score) in enumerate(ordered[:10], 1):
        print(f"{i}. {team} - {score:.6f}")
        
        
# Gráfico de variação do PR da primeira seleção
fig = plt.figure(figsize=(8,5))
values = [scores_by_d[d][0][1] for d in d_values]
plt.plot(d_values, values, marker="o")
plt.xlabel("Damping Factor (d)")
plt.ylabel("Score do Top 1")
plt.title("Impacto do Damping Factor no PageRank")
plt.tight_layout()
plt.savefig("pr_impacto_damping.png")
plt.close()