import json
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

MAX_RANK = 30
with open('boardgames_100.json') as f:
  data = json.load(f)

id2rank = {}

for game in data:
  id2rank[game['id']] = game['rank']

# Maximal Marginal Relevance
def mmr(scores, similarity_matrix, lambda_param, max_selected=10):
    n = len(scores)
    selected = []
    unselected = list(range(n))

    # Choose the item with the highest score as the first item
    first_idx = np.argmax(scores)
    selected.append(first_idx)
    unselected.remove(first_idx)

    while len(selected) < max_selected and len(unselected) > 0:
        # Calculate the similarity scores between the selected and unselected items
        similarity_scores = similarity_matrix[selected, :][:, unselected]

        # Calculate the MMR scores for each unselected item
        mmr_scores = lambda_param * scores[unselected] - (1 - lambda_param) * np.max(similarity_scores, axis=0)

        # Choose the item with the highest MMR score as the next item to add
        next_idx = unselected[np.argmax(mmr_scores)]
        selected.append(next_idx)
        unselected.remove(next_idx)

    return selected

results = []
n = len(data)
S = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            i_liked = set(data[i]["recommendations"]["fans_liked"])
            j_liked = set(data[j]["recommendations"]["fans_liked"])
            S[i][j] = len(set(i_liked).intersection(set(j_liked))) / \
                      (len(set(i_liked)) + len(set(j_liked)))


num_reviews = np.array([game['rating']['num_of_reviews'] for game in data])
scores = num_reviews / num_reviews.sum()
lambda_param = 0.5
selected_indices = mmr(scores, S, lambda_param, MAX_RANK)
for item in  [data[idx] for idx in selected_indices]:
    print('- #' + str(item['rank']),item['title'], scores[id2rank[item['id']]-1])

G = nx.DiGraph()
matrix = np.zeros((n, n))
G.add_nodes_from(np.arange(0, n))

labels = {}
for game in data:
  fans_liked = game['recommendations']['fans_liked']
  x = game['rank'] - 1
  if x in selected_indices:
    labels[x] = game['title']
  for liked in fans_liked:
    if liked in id2rank:
      y = id2rank[liked] - 1
      G.add_edge(x, y)
      matrix[x][y] = 1

pos = nx.nx_agraph.graphviz_layout(G)

for i in range(n):
  if i not in selected_indices:
     G.remove_node(i)

kmeans = KMeans(n_clusters=3)
kmeans.fit(matrix)

print(kmeans.labels_)

nx.draw(G, pos, ax=None, labels=labels, node_color=kmeans.labels_[selected_indices], with_labels=True, font_size=8)
plt.show()
