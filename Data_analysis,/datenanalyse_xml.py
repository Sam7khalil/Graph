import json
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt
from collections import Counter

# Lade und verarbeite die Daten aus der JSON-Datei
with open('boardgames_100.json') as f:
    data = json.load(f)

# Extrahiere die Spiel-IDs und die Fans-also-Like-Beziehungen
game_ids = []
fans_also_like = {}
for game in data:
    game_id = game['id']
    game_ids.append(game_id)
    fans = game['recommendations']['fans_liked']
    fans_also_like[game_id] = fans

##########################################################################
# Führe die Clusteranalyse basierend auf den Fans-also-Like-Beziehungen durch

# Erstelle die Ähnlichkeitsmatrix basierend auf den Fans-also-Like-Beziehungen
n = len(data)
similarity_matrix = np.zeros((n, n))
for i, game in enumerate(data):
    game_id = game['id']
    fans = fans_also_like[game_id]
    for fan in fans:
        fan_idx = next((idx for idx, g in enumerate(data) if g['id'] == fan), None)
        if fan_idx is not None:
            similarity_matrix[i, fan_idx] = 1

# Führe die Clusteranalyse durch
num_clusters = 2  # Anzahl der gewünschten Cluster
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(similarity_matrix)

##########################################################################
# Extrahiere die Playingtime für jedes Spiel in den Clustern

# Lade und verarbeite die Daten aus der XML-Datei
tree = ET.parse('gameinfo.xml')
root = tree.getroot()

# Erstelle eine leere Liste für die Cluster
clusters = [[] for _ in range(num_clusters)]

# Weise jedes Spiel den entsprechenden Cluster zu
missing_game_ids = []
for i, game_id in enumerate(game_ids):
    cluster_label = cluster_labels[i]

    # Suche das Spiel in der XML-Datei basierend auf der Spiel-ID
    game_element = root.find(f".//item[@id='{game_id}']")

    if game_element is not None:
        playing_time_element = game_element.find("playingtime")
        if playing_time_element is not None:
            playing_time =int(playing_time_element.get("value"))
            clusters[cluster_label].append(playing_time)
        else:
            print(f"Playingtime nicht gefunden für Spiel mit ID '{game_id}'")
    else:
        missing_game_ids.append(game_id)


# Berechne die durchschnittliche Playingtime für jeden Cluster
avg_playing_times = []
for cluster in clusters:
    if cluster:  # Überprüfe, ob der Cluster nicht leer ist
        avg_playing_time = round(np.mean(cluster))
    else:
        avg_playing_time = 0  # Setze den Durchschnitt auf 0, wenn der Cluster leer ist
    avg_playing_times.append(avg_playing_time)

# Gib die durchschnittliche Playingtime für jeden Cluster aus
for i, avg_playing_time in enumerate(avg_playing_times):
    print(f"Durchschnittliche Playingtime für Cluster {i}: {avg_playing_time} Minuten")

# Gib die fehlenden Spiel-IDs aus
for missing_game_id in missing_game_ids:
    print(f"Spiel mit ID '{missing_game_id}' nicht in der XML-Datei gefunden!")

################## Visualisierung #####################

# Erstelle ein Histogramm für jede Cluster-Spielzeit
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange', 'red', 'purple']  # Farben für die Cluster
labels = [f'Cluster {i}' for i in range(num_clusters)]  # Labels für die Cluster
plt.hist(clusters, bins=20, color=colors[:num_clusters], label=labels[:num_clusters], alpha=0.7)
plt.xlabel('Spielzeit (Minuten)')
plt.ylabel('Anzahl Spiele')
plt.title('Verteilung der Spielzeiten in jedem Cluster')
plt.legend()
#plt.show()

##########################################################################
# Extrahiere die Min-Max-Spielerzahl für jedes Spiel in den Clustern
min_max_players = [[] for _ in range(num_clusters)]

# Weise jedes Spiel den entsprechenden Cluster zu
missing_game_ids = []
for i, game_id in enumerate(game_ids):
    cluster_label = cluster_labels[i]

    # Suche das Spiel in der XML-Datei basierend auf der Spiel-ID
    game_element = root.find(f".//item[@id='{game_id}']")

    if game_element is not None:
        minplayers_element = game_element.find("minplayers")
        maxplayers_element = game_element.find("maxplayers")
        if minplayers_element is not None and maxplayers_element is not None:
            min_players = int(minplayers_element.get("value"))
            max_players = int(maxplayers_element.get("value"))
            min_max_players[cluster_label].append((min_players, max_players))
        else:
            print(f"Min-Max-Spielerzahl nicht gefunden für Spiel mit ID '{game_id}'")
    else:
        missing_game_ids.append(game_id)

# Berechne den Durchschnitt der Min-Max-Spielerzahl für jeden Cluster
avg_min_max_players = []
for cluster in min_max_players:
    if cluster:  # Überprüfe, ob der Cluster nicht leer ist
        min_players = [min_players for min_players, _ in cluster]
        max_players = [max_players for _, max_players in cluster]
        avg_min_players = round(np.mean(min_players))
        avg_max_players = round(np.mean(max_players))
    else:
        avg_min_players = 0  # Setze den Durchschnitt auf 0, wenn der Cluster leer ist
        avg_max_players = 0
    avg_min_max_players.append((avg_min_players, avg_max_players))

# Gib den Durchschnitt der Min-Max-Spielerzahl für jeden Cluster aus
for i, (avg_min_players, avg_max_players) in enumerate(avg_min_max_players):
    print(f"Durchschnittliche Min-Max-Spielerzahl für Cluster {i}: Min-Spieler: {avg_min_players}, Max-Spieler: {avg_max_players}")

# Gib die fehlenden Spiel-IDs aus
for missing_game_id in missing_game_ids:
    print(f"Spiel mit ID '{missing_game_id}' nicht in der XML-Datei gefunden!")

################## Visualisierung #####################

# Erstelle Histogramm für jedes Cluster
for i, cluster in enumerate(min_max_players):
    # Erstelle Histogramm für Min-Spieler
    plt.figure(figsize=(8, 6))
    plt.hist(min_players, bins='auto', color='steelblue', edgecolor='black')
    plt.title(f"Cluster {i} - Min-Spieler")
    plt.xlabel("Min-Spieler")
    plt.ylabel("Anzahl Spiele")
    plt.grid(True)

    # Erstelle Histogramm für Max-Spieler
    plt.figure(figsize=(8, 6))
    plt.hist(max_players, bins='auto', color='steelblue', edgecolor='black')
    plt.title(f"Cluster {i} - Max-Spieler")
    plt.xlabel("Max-Spieler")
    plt.ylabel("Anzahl Spiele")
    plt.grid(True)

#plt.show()

###############################################################################

# Erstelle eine leere Liste für die Beschreibungen in jedem Cluster
descriptions = [[] for _ in range(num_clusters)]

# Weise jeder Spielbeschreibung den entsprechenden Cluster zu
for i, game_id in enumerate(game_ids):
    cluster_label = cluster_labels[i]

    # Suche das Spiel in der XML-Datei basierend auf der Spiel-ID
    game_element = root.find(f".//item[@id='{game_id}']")

    if game_element is not None:
        description_element = game_element.find("description")
        if description_element is not None:
            description = description_element.text
            descriptions[cluster_label].append(description)

# Führe die Textanalyse für jede Cluster-Beschreibung durch
for i, cluster in enumerate(descriptions):
    print(f"Cluster {i}:")

    # Kombiniere alle Beschreibungen in einem Textkorpus
    corpus = ' '.join(cluster)

    # Tokenisierung und Entfernung von Stoppwörtern
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(corpus)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    # Häufigkeitsverteilung der Wörter
    freq_dist = FreqDist(tokens)

    # Gib die 10 häufigsten Wörter und ihre Häufigkeiten aus
    for word, frequency in freq_dist.most_common(10):
        print(f"{word}: {frequency}")
    print("\n")

################## Visualisierung #####################
"""
    # Generiere ein Wortwolken-Diagramm
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq_dist)

    # Zeige das Wortwolken-Diagramm an
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Wortwolke für Cluster {i}")
    #plt.show()
"""

###############################################################################

# Erstelle eine leere Liste für die Spieldesigner in jedem Cluster
designers = [[] for _ in range(num_clusters)]

# Weise jedem Spiel den entsprechenden Cluster zu und extrahiere die Spieldesigner
for i, game_id in enumerate(game_ids):
    cluster_label = cluster_labels[i]

    # Suche das Spiel in der XML-Datei basierend auf der Spiel-ID
    game_element = root.find(f".//item[@id='{game_id}']")

    if game_element is not None:
        designer_elements = game_element.findall(".//link[@type='boardgamedesigner']")
        if designer_elements:
            designer_names = [designer.get('value') for designer in designer_elements]
            designers[cluster_label].extend(designer_names)

# Ermittle die gemeinsamen Spieldesigner für jeden Cluster und deren Häufigkeit
for i, cluster in enumerate(designers):
    print(f"Cluster {i}:")
    print()
    
    designer_counter = Counter(cluster)
    
    if designer_counter:
        most_common_designer, count = designer_counter.most_common(1)[0]
        
        print(f"Häufigster Spieldesigner: {most_common_designer} ({count} Spiele)")
        
        print("Weitere gemeinsame Spieldesigner:")
        for designer, designer_count in designer_counter.most_common():
            if designer != most_common_designer:
                print(f"{designer} ({designer_count} Spiele)")
    else:
        print("Keine gemeinsamen Spieldesigner gefunden.")
    
    print("\n")

################## Visualisierung #####################

# Ermittle die Historie der Spieldesigner in jedem Cluster
designer_history = [[] for _ in range(num_clusters)]
for i, cluster in enumerate(designers):
    designer_counter = Counter(cluster)
    for designer, count in designer_counter.items():
        if count > 1:  # Filtere Designer, die nur einmal vorkommen
            designer_history[i].append((designer, count))

# Erstelle das Balkendiagramm der Historie der Spieldesigner in jedem Cluster
for i, cluster_history in enumerate(designer_history):
    designers = [designer for designer, _ in cluster_history]
    counts = [count for _, count in cluster_history]
    
    plt.figure(figsize=(10, 6))
    plt.bar(designers, counts)
    plt.title(f'Historie der Spieldesigner in Cluster {i}')
    plt.xlabel('Spieldesigner')
    plt.ylabel('Anzahl')
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()








