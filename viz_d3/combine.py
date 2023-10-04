#!/usr/bin/env python

from pathlib import Path
import json
import networkx as nx
import networkx.algorithms as nxalg
import numpy as np
from sklearn.cluster import KMeans

with Path("../cat.json").open() as f:
    cat = json.load(f)

with Path("../boardgames_100.json").open() as f:
    boardgames = json.load(f)

CLUSTER_COUNT = 4
GAME_COUNT = 100

boardgames = boardgames[:GAME_COUNT]
cat = cat[:GAME_COUNT]

game_id2idx = {game["id"]: i for i, game in enumerate(boardgames)}

liked_by = {}
for i, game in enumerate(boardgames):
    game_id = game_id2idx[game['id']]
    fans = game['recommendations']['fans_liked']
    for fan in fans:
        if fan not in game_id2idx:
            continue
        idx = game_id2idx[fan]
        liked_by.setdefault(idx, set())
        liked_by[idx].add(game_id)

n = len(boardgames)
similarity_matrix = np.zeros((n, n))
for i, game in enumerate(boardgames):
    game_id = game['id']
    liked_me = liked_by[i] if i in liked_by else set()
    for liked in game['recommendations']['fans_liked']:
        if liked not in game_id2idx:
            continue
        liked_id = game_id2idx[liked]
        similarity_matrix[i, liked_id] = len(liked_me & liked_by[liked_id])
kmeans = KMeans(n_clusters=CLUSTER_COUNT, random_state=0, n_init='auto').fit(similarity_matrix)

for cat, bg in zip(cat, boardgames):
    cat_poll_vector = np.array(
        list(
            map(
                lambda item: item["votes"],
                sorted(cat["poll_results"], key=lambda x: x["category"]["id"]),
            )
        )
    )
    cat_poll_vector = cat_poll_vector / np.linalg.norm(cat_poll_vector)
    category_poll = {
        "main": cat["main_categories"][0],
        "results": cat["poll_results"],
        "vector": list(cat_poll_vector),
    }
    bg["category_poll"] = category_poll
    bg["fans_liked"] = list(
        filter(lambda x: x in game_id2idx, bg["recommendations"]["fans_liked"])
    )
    del bg["recommendations"]


G = nx.DiGraph()
for game in boardgames:
    G.add_node(game["id"])
    for liked in game["fans_liked"]:
        G.add_edge(game["id"], liked)


edge_centrality = nxalg.edge_betweenness_centrality(G, normalized=True)

betweenness_centrality = nxalg.betweenness_centrality(G, normalized=True)
closeness_centrality = nxalg.closeness_centrality(G)
eigen_centrality = nxalg.eigenvector_centrality(G)
harmonic_centrality = nxalg.harmonic_centrality(G)
load_centrality = nxalg.load_centrality(G)
pagerank_centrality = nxalg.pagerank(G)

for i, game in enumerate(boardgames):
    game["score"] = {
        "betweenness": betweenness_centrality[game["id"]],
        "closeness": closeness_centrality[game["id"]],
        "eigen": eigen_centrality[game["id"]],
        "harmonic": harmonic_centrality[game["id"]],
        "load": load_centrality[game["id"]],
        "pagerank": pagerank_centrality[game["id"]],
    }
    game["cluster"] = int(kmeans.labels_[i])
    game["fans_liked"] = list(
        map(
            lambda liked: {
                "score": edge_centrality[(game["id"], liked)],
                "cat_score": np.dot(
                    game["category_poll"]["vector"],
                    boardgames[game_id2idx[liked]]["category_poll"]["vector"],
                ),
                "id": liked,
                "reciprocal": ((liked, game["id"]) in G.edges),
                "same_category": game["category_poll"]["main"]
                == boardgames[game_id2idx[liked]]["category_poll"]["main"],
            },
            game["fans_liked"],
        ),
    )


with Path("combined.json").open("w") as f:
    json.dump(boardgames, f, indent=2)
