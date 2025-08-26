## Objetivo

Realizar uma análise exploratória e aplicar um modelo de árvore de decisão utilizando a base de dados **FIFA World Ranking (1993–2018)** disponível no Kaggle. O foco está em compreender o comportamento dos rankings das seleções ao longo do tempo e avaliar a capacidade de um modelo simples em prever a faixa de posição das seleções.


## Montagem do Roteiro

Os pontos "tarefas" são os passos que foram seguidos para a realização do roteiro. Cada tarefa apresenta evidências claras de execução (estatísticas, gráficos, código e resultados).


### Tarefa 1 - Exploração dos Dados

Foi utilizado o dataset contendo os rankings oficiais da FIFA de 1993 até 2024.

Colunas principais:
- **rank**: posição da seleção no ranking
- **country_full**: nome da seleção
- **country_abrv**: abreviação de 3 letras
- **total_points**: pontos acumulados
- **previous_points**: pontos da edição anterior
- **rank_change**: variação de posição em relação ao ranking anterior
- **confederation**: confederação (UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC)
- **rank_date**: data do ranking

Estatísticas descritivas:
- `rank`: varia entre 1 e mais de 200, média próxima de 90
- `total_points`: entre ~700 e 2000 pontos
- `rank_change`: geralmente entre -5 e +5
- `confederation`: maior número de seleções pertencem à UEFA

#### Visualizações

Distribuição dos ranks das seleções:

![Distribuição do Rank](distribuicao_rank.png)

Evolução do ranking de Brasil, Alemanha e Argentina:

![Evolução do Ranking](evolucao_ranking.png)

---

### Tarefa 2 - Pré-processamento

- Remoção de valores ausentes  
- Conversão da variável *confederation* em dummies (one-hot encoding)  
- Criação da variável-alvo `faixa_rank`, agrupando posições em 5 faixas:  
  - 1 a 50  
  - 51 a 100  
  - 101 a 150  
  - 151 a 200  
  - acima de 200 

## Tarefa 3 - Divisão dos Dados

Separação em conjuntos:
- **Treino**: 80%  
- **Teste**: 20%

### Tarefa 4 - Treinamento do Modelo

O treinamento foi realizado com o algoritmo **Decision Tree Classifier**, utilizando como variáveis de entrada os pontos da edição anterior, a variação no ranking e a confederação de cada seleção. O modelo foi ajustado com a base de treino, sem alterações complexas de parâmetros, para manter a simplicidade do experimento.


### Tarefa 5 - Avaliação do Modelo

A avaliação foi feita com a base de teste. O desempenho alcançado foi uma acurácia em torno de 70%, mostrando que o modelo conseguiu identificar padrões gerais, especialmente ao diferenciar as seleções de elite das demais. A matriz de confusão evidenciou que as classes mais representadas (1–100) tiveram maior acerto, enquanto as classes menos frequentes apresentaram confusão maior. O relatório de classificação confirmou essa tendência, apontando precisão mais elevada para as seleções do topo do ranking.  

## Questionário, Projeto ou Plano

Não será necessário neste roteiro.

## Discussões

O processo de exploração inicial foi relativamente simples, assim como a geração das estatísticas descritivas. A etapa mais desafiadora foi lidar com a categorização do ranking, pois prever a posição exata de cada seleção se mostrou inviável com um modelo de árvore de decisão básico. A solução adotada foi agrupar as seleções em faixas de ranking, o que permitiu que o modelo tivesse resultados mais consistentes.


## Conclusão

Foi possível concluir que o modelo de árvore de decisão consegue identificar padrões básicos no ranking da FIFA, principalmente ao diferenciar seleções de elite das demais. No entanto, o desempenho ainda apresenta limitações quando aplicado a classes menos representadas. Como sugestões de melhoria, seria interessante incluir variáveis adicionais relacionadas ao desempenho esportivo (como vitórias, gols e torneios disputados) e manter o dataset atualizado com os novos rankings divulgados pela FIFA.

