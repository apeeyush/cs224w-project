'''
Parser for citation network data. 
Dataset is available at https://aminer.org/citation, Citation network V1.
'''


import networkx as nx


knownInvalidAuthorNames = set()
knownInvalidAuthorNames.add(" II")
knownInvalidAuthorNames.add(" III")
knownInvalidAuthorNames.add(" IV")
knownInvalidAuthorNames.add(" V")
knownInvalidAuthorNames.add(" VI")
knownInvalidAuthorNames.add(" Jr.")
knownInvalidAuthorNames.add("Staff")

pTest = 0.01 # [0, 0.01] falls into test set
pVal = 0.02 # (0.01, 0.02] falls into validation set


def appendNode(G, node):
  '''
  Add node to G if it's not already there
  Optionally adding 'val' and 'test' attributes to node with set probability.
  '''
  if not G.has_node(node):
    G.add_node(node)


def loadGraph(fileName):
  G = nx.MultiGraph()
  authorMap = dict() # key: author, value: list of paperId
  publicationMap = dict() # key: publication+year, value: list of paperId
  blacklist = set()
  # In each iteration we keep the following information until we see the new line
  # authors: list of authors
  # references: list of references
  # currentPaperId
  # currentPaperTitle
  # publicationYear
  # publicationVenue
  references = []
  count = 0
  with open(fileName, 'r') as f:
    for line in f.readlines():
      prefix = line[:2]
      if prefix == "#*":
        currentPaperTitle = line[2:].strip()
      elif prefix == "#@":
        authorsRaw = line[2:].strip()
      elif prefix == "#t":
        publicationYear = line[2:].strip()
      elif prefix == "#c":
        publicationVenue = line[2:].strip()
      elif prefix == "#i":
        currentPaperId = int(line[6:])
      elif prefix == "#%":
        references.append(int(line[2:]))
      elif prefix == "\n":
        if len(references) < 1 or currentPaperId in blacklist:
          blacklist.add(currentPaperId)
        else:
          appendNode(G, currentPaperId)
          for reference in references:
            if reference not in blacklist:
              appendNode(G, reference)
              G.add_edge(currentPaperId, reference, type="reference")
          if len(authorsRaw) >= 1:
            authors = authorsRaw.split(',')
            for author in authors:
              if author not in knownInvalidAuthorNames:
                authorMap.setdefault(author,[]).append(currentPaperId)
          hashKey = publicationVenue + publicationYear
          publicationMap.setdefault(hashKey,[]).append(currentPaperId)

        currentPaperTitle, authorsRaw, publicationYear, publicationVenue, currentPaperId = "", "", "", "", ""
        references = []
  print G.number_of_nodes(), G.number_of_edges()
  annotateGraphWithEdges(G, authorMap, "coauthor")
  print G.number_of_nodes(), G.number_of_edges()
  annotateGraphWithEdges(G, publicationMap, 'publication')
  print G.number_of_nodes(), G.number_of_edges()
  return G


def annotateGraphWithEdges(G, map, t):
  '''
  map: a dictionary of <key, list[paperId]>
  For each key in the map, add pair-wise edge to every node denoted by value 
  '''
  for k in map:
    papers = map[k]
    for index1 in xrange(len(papers)):
      for index2 in xrange(index1+1, len(papers)):
        G.add_edge(papers[index1], papers[index2], type=t)


G = loadGraph("outputacm.txt")
