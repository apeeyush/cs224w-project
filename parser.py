'''
Parser for citation network data. 
Dataset is available at https://aminer.org/citation, Citation network V1.
'''
import networkx as nx
import random

random.seed(0)

knownInvalidAuthorNames = set()
knownInvalidAuthorNames.add(" II")
knownInvalidAuthorNames.add(" III")
knownInvalidAuthorNames.add(" IV")
knownInvalidAuthorNames.add(" V")
knownInvalidAuthorNames.add(" VI")
knownInvalidAuthorNames.add(" Jr.")
knownInvalidAuthorNames.add("Staff")

pTest = 0.05 # [0, 0.01] falls into test set
pVal = 0.1 # (0.01, 0.02] falls into validation set
enableAttributes = True

# Flags to control whether output graph G includes following types of edges
ingestionFlags = {
  "reference": True,
  "coauthor": True,
  "publication": True
}

def appendNode(G, node):
  '''
  Add node to G if it's not already there
  Optionally adding 'val' and 'test' attributes to node with set probability.
  '''
  if not G.has_node(node):
    if enableAttributes:
      r = random.uniform(0, 1)
      if r <= pTest:
        G.add_node(node, test=True)
        return
      elif r <= pVal:
        G.add_node(node, val=True)
        return
    G.add_node(node)
      

def appendEdge(G, node1, node2, label):
  shouldIngest = ingestionFlags.get(label, False)
  if shouldIngest:
    G.add_edge(node1, node2, type=label)


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
              appendEdge(G, currentPaperId, reference, "reference")
          if len(authorsRaw) >= 1:
            authors = authorsRaw.split(',')
            for author in authors:
              if author not in knownInvalidAuthorNames:
                authorMap.setdefault(author,[]).append(currentPaperId)
          hashKey = publicationVenue + publicationYear
          publicationMap.setdefault(hashKey,[]).append(currentPaperId)

        currentPaperTitle, authorsRaw, publicationYear, publicationVenue, currentPaperId = "", "", "", "", ""
        references = []
  print "After adding reference edges:"
  printGraphStat(G)
  annotateGraphWithEdges(G, authorMap, "coauthor")
  print "After adding coauthor edges:"
  printGraphStat(G)
  annotateGraphWithEdges(G, publicationMap, 'publication')
  print "After adding publication edges:"
  printGraphStat(G)
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
        appendEdge(G, papers[index1], papers[index2], t)


def printGraphStat(G):
  print G.number_of_nodes(), G.number_of_edges()


def printNodeStat(G):
  numVal = 0
  numTest = 0
  numTrain = 0
  for node in G.__iter__():
    if "val" in G.nodes[node]:
      numVal += 1
    elif "test" in G.nodes[node]:
      numTest += 1
    else:
      numTrain += 1
  print("Number of nodes labeled as val, test, train are %d, %d, %d"
    % (numVal, numTest, numTrain))


G = loadGraph("outputacm.txt")
printNodeStat(G)
