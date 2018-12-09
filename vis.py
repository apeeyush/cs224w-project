from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
import snap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
from time import time

perplexities = [5, 30, 50, 100] # Try different values
embeds = np.load ("coauthor.npy") # Or collaborations
nodes = len(embeds)
plt_cnt = int(nodes/200) # Take 0.5% of nodes
embeds2 = [] # visualize this list

# Pick random nodes
for i in range(plt_cnt):
  node = int(random.random() * nodes)
  embeds2.append (embeds[node])

# Visualize
for i, perp in enumerate(perplexities):
  t0 = time()
  Y = manifold.TSNE(n_components=2, perplexity=perp).fit_transform(embeds2)
  t1 = time()
  print "perplexity=%d in %.2g sec" % (perp, t1 - t0)
  plt.clf()
  plt.scatter(Y[:, 0], Y[:, 1])
  plt.show()
  file_name = "coauthor_" + str(i) + ".png" # or collaboration
  plt.savefig(file_name)
