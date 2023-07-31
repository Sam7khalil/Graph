import json
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

MAX_RANK = 10
with open('boardgames_100.json') as f:
  data = json.load(f)

id2rank = {}

for game in data:
  id2rank[game['id']] = game['rank']

# PageRank
results = []
n = len(data)
A = np.zeros((n, n))
for i, game in enumerate(data):
    links =  game['recommendations']['fans_liked']
    outgoing_links = len(links)
    for j in links:
        if j in id2rank:
            A[i][id2rank[j]-1] = 1/outgoing_links
v = np.ones(n) / n
last_v = np.ones(n) / n

damping_factor=0.85
max_iterations=100
convergence_threshold=0.0001


num_reviews = np.array([game['rating']['num_of_reviews'] for game in data])
ranks = num_reviews / num_reviews.sum()
for i in range(max_iterations):
    last_v = v.copy()
    v = damping_factor * np.matmul(A, v) + (1 - damping_factor) # * ranks
    if np.abs(v - last_v).sum() < convergence_threshold:
        break

ratings = np.argsort(-v)

for item in  [data[idx] for idx in ratings[:MAX_RANK]]:
    print('- #' + str(item['rank']),item['title'], v[id2rank[item['id']]-1])


G = nx.DiGraph()
matrix = np.zeros((n, n))
G.add_nodes_from(np.arange(0, n))

labels = {}
for game in data:
  fans_liked = game['recommendations']['fans_liked']
  x = game['rank'] - 1
  if x in ratings[:MAX_RANK]:
    labels[x] = game['title']
  for liked in fans_liked:
    if liked in id2rank:
      y = id2rank[liked] - 1
      G.add_edge(x, y, weight=(ratings[x] + ratings[y])/2)
      matrix[x][y] = 1

pos = nx.nx_agraph.graphviz_layout(G, "neato")

for i in range(n):
  if i not in ratings[:MAX_RANK]:
     G.remove_node(i)

kmeans = KMeans(n_clusters=3)
kmeans.fit(matrix)

print(kmeans.labels_)

nx.draw(G, pos, ax=None, labels=labels, node_color=kmeans.labels_[ratings[:MAX_RANK]], with_labels=True, font_size=8)
plt.savefig('pagerank.png', dpi=300)
plt.show()
