import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Lade und verarbeite die Daten aus der JSON-Datei
with open('boardgames_100.json') as f:
    data = json.load(f)

##########################################################################
# Führe die Clusteranalyse basierend auf den Fans-also-Like-Beziehungen durch

# Extrahiere die Fans-also-Like-Beziehungen
fans_also_like = {}
for game in data:
    game_id = game['id']
    fans = game['recommendations']['fans_liked']
    fans_also_like[game_id] = fans

# Erstelle die Ähnlichkeitsmatrix basierend auf den Fans-also-Like-Beziehungen
n = len(data)
similarity_matrix = np.zeros((n, n))
for i, game in enumerate(data):
    game_id = game['id']
    fans = fans_also_like[game_id]
    for fan in fans:
        fan_idx = next((idx for idx, g in enumerate(data) if g['id'] == fan), None)
        if fan_idx is not None:
            similarity_matrix[i, fan_idx] = 1 # Jeder Eintrag muss 1 sein, da jedes Spiel im Datensatz einen nicht leeren 'fans_liked'-Abschnitt hat

# Führe die Clusteranalyse durch
num_clusters = 2  # Anzahl der gewünschten Cluster
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(similarity_matrix)

# Drucke die Clusterzuordnung
for i, game in enumerate(data):
    game_id = game['id']
    cluster_label = cluster_labels[i]
    print(f"Spiel '{game_id}' befindet sich im Cluster {cluster_label}")

cluster_counts = [0] * num_clusters  # Initialisiere eine Liste mit Nullen für die Cluster-Zähler

for label in cluster_labels:
    cluster_counts[label] += 1

for cluster_label, count in enumerate(cluster_counts):
    print(f"Cluster {cluster_label} enthält {count} Spiele")

##########################################################################
# Erstelle das Netzwerk basierend auf den Clusterergebnissen
"""
# Erstelle einen leeren gerichteten Graphen
G = nx.DiGraph()

# Füge Knoten für jedes Spiel hinzu
for game in data:
    game_id = game['id']
    G.add_node(game_id)

# Füge Kanten basierend auf Clusterzuordnung hinzu
for i, game in enumerate(data):
    game_id = game['id']
    cluster_label = cluster_labels[i]
    fans = game['recommendations']['fans_liked']
    for fan in fans:
        fan_idx = next((idx for idx, g in enumerate(data) if g['id'] == fan), None)
        if fan_idx is not None:
            fan_id = data[fan_idx]['id']
            G.add_edge(game_id, fan_id)

# Positioniere die Knoten im Graphen
pos = nx.spring_layout(G)

# Zeichne den Graphen
nx.draw(G, pos, with_labels=True, node_color=cluster_labels, cmap='viridis')
#plt.show()

"""
import plotly.graph_objects as go

G = nx.DiGraph()

for game in data:
    game_id = game['id']
    game_title = game['title']
    G.add_node(game_id, label=game_title)

for i, game in enumerate(data):
    game_id = game['id']
    cluster_label = cluster_labels[i]
    fans = game['recommendations']['fans_liked']
    for fan in fans:
        fan_idx = next((idx for idx, g in enumerate(data) if g['id'] == fan), None)
        if fan_idx is not None:
            fan_id = data[fan_idx]['id']
            G.add_edge(game_id, fan_id)

pos = nx.spring_layout(G)

node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers+text',
    hovertemplate='%{text}',
    textposition='top center',
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        reversescale=True,
        color=cluster_labels,
        size=10,
        colorbar=dict(
            thickness=15,
            title='Cluster Labels',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

for node in G.nodes():
    x, y = pos[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['text'] += tuple([G.nodes[node]['label']])

edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Board Game Recommendations',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600
                ))

fig.update_layout(
    annotations=[
        go.layout.Annotation(
            x=pos[node][0],
            y=pos[node][1],
            text=node,
            showarrow=False,
            font=dict(size=8)
        ) for node in G.nodes()
    ]
)

fig.show()

##########################################################################
# Kombiniere die Ergebnisse und identifiziere interessante Zusammenhänge

# Betrachte die Cluster aus der Clusteranalyse
cluster_labels_fans = kmeans.labels_

# Untersuche die Kategorien und Ränge in den Clustern
category_counts = []
average_ranks = []
for cluster_id in range(num_clusters):
    cluster_categories = []
    cluster_ranks = []
    for i in range(len(data)):
        if cluster_labels_fans[i] == cluster_id:
            cluster_categories.extend(data[i]['types']['categories'])
            cluster_ranks.append(data[i]['rank'])
    category_counts.append({category['name']: cluster_categories.count(category['name']) for category in cluster_categories})
    average_ranks.append(np.mean(cluster_ranks))

# Untersuche die häufigste Mechanik in den Clustern
cluster_mechanics = []
for cluster_id in range(num_clusters):
    cluster_games = [data[i] for i in range(len(data)) if cluster_labels_fans[i] == cluster_id]
    all_mechanics = [mechanic['name'] for game in cluster_games for mechanic in game['types']['mechanics']]
    most_common_mechanic = max(set(all_mechanics), key=all_mechanics.count)
    cluster_mechanics.append(most_common_mechanic)

# Drucke die Ergebnisse

category_counts = []
average_ranks = []
    
for cluster_id in range(num_clusters):
    cluster_categories = []
    cluster_ranks = []
        
    for i in range(len(data)):
        if cluster_labels_fans[i] == cluster_id:
            cluster_categories.extend(data[i]['types']['categories'])
            cluster_ranks.append(data[i]['rank'])
                
    category_count = {}
    for category in cluster_categories:
        name = category['name']
        if name in category_count:
            category_count[name] += 1
        else:
            category_count[name] = 1

    category_counts.append(category_count)
    average_ranks.append(np.mean(cluster_ranks))

for i, category_count in enumerate(category_counts):
    max_count = max(category_count.values())
    most_common_categories = [category for category, count in category_count.items() if count == max_count]
        
    print(f"Cluster {i+1}:")
    print(f"  Anzahl Spiele: {cluster_labels_fans.tolist().count(i)}")
    print("  Most common category:", most_common_categories)
    print(f"  Durchschnittlicher Rang: {average_ranks[i]}")
    print(f"  Häufigste Mechanik: {cluster_mechanics[i]}")
    print()

###############################################################################
# Erstelle Visualisierungen der Zusammenhänge/Ergebnisse

#Word Cloud of Most Common Categories in Each Cluster:
for cluster_id, category_count in enumerate(category_counts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(category_count)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Cluster {cluster_id + 1}: Most Common Categories")
    plt.axis('off')
    plt.show()

    # Create a bar plot of average ranks for each cluster
plt.figure(figsize=(8, 6))
plt.bar(range(num_clusters), average_ranks)
plt.xticks(range(num_clusters), labels=[f"Cluster {i + 1}" for i in range(num_clusters)])
plt.xlabel('Clusters')
plt.ylabel('Average Rank')
plt.title('Average Rank by Cluster')
plt.show()

#####################################
# Erstelle Balkendiagramme für die Häufigkeit der Kategorien in den Clustern

for i in range(num_clusters):
    categories = list(category_counts[i].keys())
    counts = list(category_counts[i].values())

    # Sammle alle Kategorien aus den Clustern
    all_categories = set()
    for j in range(num_clusters):
        all_categories.update(category_counts[j].keys())
    # Fehlende Kategorien auffüllen
    missing_categories = [category for category in all_categories if category not in categories]
    missing_counts = [0] * len(missing_categories)
    categories += missing_categories
    counts += missing_counts

    # Sortieren nach Kategorienamen
    sorted_indices = np.argsort(categories)
    categories = [categories[idx] for idx in sorted_indices]
    counts = [counts[idx] for idx in sorted_indices]

    # Erstelle eine neue Figur für jedes Cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(categories, counts)
    ax.set_title(f'Häufigkeit der Kategorien in Cluster {i+1}')
    ax.set_xticks(np.arange(len(categories)))  # Manuell die Positionen der Tickmarks festlegen
    ax.set_xticklabels(categories, rotation=90)  # Rotation um 90 Grad
    ax.set_xlabel('Kategorie')
    ax.set_ylabel('Anzahl')

"""
#category_counts-Liste Ausgabe 
for i, category_count in enumerate(category_counts):
    print(f"Cluster {i+1}:")
    for category, count in category_count.items():
        print(f"  Kategorie: {category}, Anzahl Spiele: {count}")
"""

##############################

# Erstelle Balkendiagramme für die Häufigkeit der Mechaniken in den Clustern

cluster_mechanics = [[] for _ in range(num_clusters)]

# Sammle die Mechaniken für jedes Cluster
for i, game in enumerate(data):
    game_id = game['id']
    cluster_label = cluster_labels[i]
    mechanics = game['types']['mechanics']
    for mechanic in mechanics:
        cluster_mechanics[cluster_label].append(mechanic['name'])

# Zähle die Mechaniken in den Clustern
mechanics_counts = [pd.Series(cluster).value_counts().to_dict() for cluster in cluster_mechanics]

# Erstelle das Balkendiagramm für die Häufigkeit der Mechaniken in den Clustern
for i in range(num_clusters):
    mechanics = list(mechanics_counts[i].keys())
    counts = list(mechanics_counts[i].values())

    # Sortieren nach Mechaniknamen
    sorted_indices = np.argsort(mechanics)
    mechanics = [mechanics[idx] for idx in sorted_indices]
    counts = [counts[idx] for idx in sorted_indices]

    # Erstelle eine neue Figur für jedes Cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(mechanics, counts)
    ax.set_title(f'Häufigkeit der Mechaniken in Cluster {i+1}')
    ax.set_xticks(np.arange(len(mechanics)))  # Manuell die Positionen der Tickmarks festlegen
    ax.set_xticklabels(mechanics, rotation=90)  # Rotation um 90 Grad
    ax.set_xlabel('Mechanik')
    ax.set_ylabel('Anzahl')

##############################
# Erstelle eine leere Liste von Rängen für jedes Cluster
cluster_ranks = [[] for _ in range(num_clusters)]

# Fülle die Liste der Ränge für jedes Cluster
for i, game in enumerate(data):
    game_id = game['id']
    cluster_label = cluster_labels[i]
    rank = game['rank']
    cluster_ranks[cluster_label].append(rank)

# Erstelle Boxplots für die Ränge in den Clustern
fig, ax = plt.subplots(figsize=(8, 6))
data_per_cluster = cluster_ranks
labels_per_cluster = [f'Cluster {i+1}' for i in range(num_clusters)]
ax.boxplot(data_per_cluster, labels=labels_per_cluster)
ax.set_title('Rang in den Clustern')
ax.set_xlabel('Cluster')
ax.set_ylabel('Rang')

plt.tight_layout()
plt.show()


