"""
Initial experiments with category data
"""


import json
from pathlib import Path
import pygraphviz as pgv
import pandas as pd
from scipy.spatial import distance
import numpy as np
import subprocess

with Path('cat.json').open() as f:
    cat_info = json.load(f)[:40]

with Path('boardgames_100.json').open() as f:
    bg = json.load(f)[:40]

games_in_category = {}
cat_info_by_id = {}
for g in cat_info:
    cat_info_by_id[g['id']] = g
    winners = [p['name'] for p in g['main_categories']]

    for w in winners:
        games_in_category.setdefault(w, [])
        games_in_category[w].append(g['id'])


all_cats = set()
all_mechs = set()
all_designers = set()
all_likes = set()
n = len(bg)

id2idx = {}
idx2id = {}

for game in bg:
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

G = pgv.AGraph(directed=True, overlap_scaling=3)
for game in bg:
    G.add_node(game['id'], label=game['title'], shape='box', width=1-game['rank']/40, height=1-game['rank']/40)
    my_cat = cat_info_by_id[game['id']]['main_categories'][0]['name']
    for liked in game['liked']:
        if liked in cat_info_by_id:
            G.add_edge(game['id'], liked, dist=1 if liked in games_in_category[my_cat] else 2)

for category, games in games_in_category.items():
    G.add_subgraph(games, name="cluster_" + category, label=category)
G.draw("subgraph.png", prog="fdp")
G.write("subgraph.gv")

positioned = subprocess.run(["fdp", "-Goverlap=prism"], input=G.string(), capture_output=True, check=True, text=True).stdout
output = subprocess.run(["gvmap", "-b", "1", "-D", "-e"], input=positioned, capture_output=True, check=True, text=True).stdout
subprocess.run(["neato", "-Tpng", "-o", "gvmap.png"], input=output,check=True, text=True)

