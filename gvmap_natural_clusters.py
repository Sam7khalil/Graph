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

game_ids = [game['id'] for game in bg]


G = pgv.AGraph(directed=True, overlap_scaling=3)
for game in bg:
    G.add_node(game['id'], label=game['title'], shape='box', width=1-game['rank']/40, height=1-game['rank']/40)

    for liked in game['recommendations']['fans_liked']:
        if liked in game_ids:
            G.add_edge(game['id'], liked)

G.draw("subgraph_alt.png", prog="fdp")

positioned = subprocess.run(["fdp", "-Goverlap=prism", "subgraph_alt.dot"], input=G.string(), capture_output=True, check=True, text=True).stdout
output = subprocess.run(["gvmap", "-e"], input=positioned, capture_output=True, check=True, text=True).stdout
subprocess.run(["neato", "-Tpng", "-o", "gvmap_alt.png"], input=output,check=True, text=True)

