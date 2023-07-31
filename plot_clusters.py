
import json
from pathlib import Path
import pygraphviz as pgv

MAX_RANK = 40

with Path('combined.json').open() as f:
    data = json.load(f)[:MAX_RANK]

id2idx = {}
idx2id = {}
games_by_cluster = {}

for i, game in enumerate(data):
    id2idx[game['id']] = i
    idx2id[i] = game['id']
    games_by_cluster.setdefault(game['cluster'], set())
    games_by_cluster[game['cluster']].add(game['id'])


G = pgv.AGraph(directed=True, overlap_scaling=3)
for game in data:
    G.add_node(game['id'], label=game['title'], shape='box', width=1-game['rank']/40, height=1-game['rank']/40)
    my_cluster = game['cluster']
    for liked in game['fans_liked']:
        if liked['id'] not in id2idx:
            continue
        same_cluster = liked['id'] in games_by_cluster[my_cluster]
        G.add_edge(game['id'], liked['id'], dist=1 if same_cluster else 2)

cluster_count = max([game['cluster'] for game in data]) + 1

for category, games in games_by_cluster.items():
    G.add_subgraph(games, name=f"cluster_{category}", label=category)
G.draw("plot.png", prog="fdp")
G.write("plot.gv")
