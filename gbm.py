"""
This script trains a gradient boosting machine (GBM) model on the BoardGameGeek dataset.
It can rank the items based on the predicted scores and may be usable as a comparison method.
"""


import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx

MAX_RANK = 30
with open('boardgames_100.json') as f:
  data = json.load(f)

all_cats = set()
all_mechs = set()
all_designers = set()
all_likes = set()
n = len(data)

id2idx = {}
idx2id = {}

for game in data:
    id2idx[game['id']] = game['rank'] -1
    idx2id[game['rank'] -1] = game['id']
    game['cat'] = [cat['id'] for cat in game['types']['categories']]
    game['mech'] = [mech['id'] for mech in game['types']['mechanics']]
    game['designer'] = [designer['id'] for designer in game['credit']['designer']]
    game['liked'] = [like for like in game['recommendations']['fans_liked']]
    game['num_of_reviews'] = game['rating']['num_of_reviews']
    game['rating'] = game['rating']['rating']

    del game['types']
    del game['credit']
    del game['recommendations']

    all_cats.update(game['cat'])
    all_mechs.update(game['mech'])
    all_designers.update(game['designer'])
    all_likes.update(game['liked'])
    all_likes.add(game['id'])

df = pd.DataFrame(data)

for category in all_cats:
    df[f"cat_{category}"] = [category in categories for categories in df["cat"]]

for mechanic in all_mechs:
    df[f"mech_{mechanic}"] = [mechanic in mechanics for mechanics in df["mech"]]

for designer in all_designers:
    df[f"designer_{designer}"] = [designer in designers for designers in df["designer"]]

for like in all_likes:
    df[f"liked_{like}"] = [like in likes for likes in df["liked"]]

X  = df.drop(columns=['cat', 'rank', 'mech', 'designer', 'liked', 'rank', 'title'], axis=1)
y = df['id']

for column in X.columns:
    X[column] = (X[column] - X[column].mean()) / X[column].std()

# Train the GBM model
params = {"objective": "reg:squarederror", "eval_metric": "rmse"}
model = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=1000)

# Calculate the predicted scores for the items
X_pred = df.drop(['cat', 'rank','mech', 'designer', 'liked', 'rank', 'title'], axis=1)
y_pred = model.predict(xgb.DMatrix(X_pred))

# Combine the predicted scores and the item IDs into a DataFrame
results = pd.DataFrame({"id": df["id"], "score": y_pred})

# Sort the items by their scores in descending order and return the top 10
selected_items = results.sort_values("score", ascending=False).head(10)["id"].tolist()

for item in  [data[id2idx[idx]] for idx in selected_items]:
   print('- #' + str(item['rank']),item['title'], item['rating'])

G = nx.DiGraph()
matrix = np.zeros((n, n))
G.add_nodes_from(np.arange(0, n))

print('sel',selected_items)

labels = {}
for game in data:
  x = id2idx[game['id']]
  if game['id'] in selected_items:
    labels[x] = game['title']
  for liked in game['liked']:
    if liked in id2idx:
      y = id2idx[liked]
      G.add_edge(x, y)
      matrix[x][y] = 1

pos = nx.nx_agraph.graphviz_layout(G, "neato")

for i in range(n):
  if idx2id[i] not in selected_items:
     G.remove_node(i)

nx.nx_agraph.write_dot(G, 'gb.dot')

kmeans = KMeans(n_clusters=3)
kmeans.fit(matrix)

print(kmeans.labels_)

selected_ids = list(map(lambda i: id2idx[i], selected_items))
print(kmeans.labels_[selected_ids])

nx.draw(G, pos, ax=None, labels=labels, node_color=kmeans.labels_[selected_ids], with_labels=True, font_size=8)
plt.savefig('gb.png', dpi=300)
plt.show()
