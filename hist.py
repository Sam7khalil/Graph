import json
import numpy as np

with open('boardgames_100.json') as f:
    data = json.load(f)

min_players = np.zeros(8)
max_players = np.zeros(8)
min_age = np.zeros(18)

for game in data:
    min_players[game['minplayers'] - 1] += 1
    max_players[game['maxplayers'] - 1] += 1
    min_age[game['minage'] - 1] += 1

xs = np.arange(1, 9)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

ax1.bar(xs, min_players, width=0.5, color='b', align='center')
ax2.bar(xs, max_players, width=0.5, color='r', align='center')
print(min_age)
print(min_age[7:])
ax3.bar(np.arange(8, len(min_age) + 1), min_age[7:], width=0.5, color='g', align='center')

ax1.set_title('Min Players')
ax2.set_title('Max Players')
ax3.set_title('Min Age')

ax1.set_xlabel('Number of Players')
ax2.set_xlabel('Number of Players')
ax3.set_xlabel('Min Age')

ax1.set_ylabel('Number of Games')
ax2.set_ylabel('Number of Games')
ax3.set_ylabel('Number of Games')

fig1.savefig('players_min.png', dpi=300)
fig2.savefig('players_max.png', dpi=300)
fig3.savefig('min_age.png', dpi=300)

