'''Parser for citation network data. 

Dataset is available at https://aminer.org/citation, Citation network V1.

Usage:
ingestionFlags = {
  "reference": True,
  "coauthor": True,
  "publication": True
}
G = parser.loadGraph("outputacm.txt", ingestionFlags)
'''

import networkx as nx
import numpy as np
import random
import json
from collections import defaultdict
from networkx.readwrite import json_graph
import pickle

random.seed(0)

knownInvalidAuthorNames = set()
knownInvalidAuthorNames.add(" II")
knownInvalidAuthorNames.add(" III")
knownInvalidAuthorNames.add(" IV")
knownInvalidAuthorNames.add(" V")
knownInvalidAuthorNames.add(" VI")
knownInvalidAuthorNames.add(" Jr.")
knownInvalidAuthorNames.add("Staff")

pTest = 0.1  # [0, 0.1] falls into test set
pVal = 0  # no validation node set
enableAttributes = True

# Only ingest x% of nodes
samplingRate = 100

paperId2NodeId = dict()
currentNodeId = 0

pEdgeVal = 0.01 # 1% of edges are hidden from generated graph as validation edges
validationEdges = set()

def appendNode(G, paperId):
  '''
  Add node to G if it's not already there
  Optionally adding 'val' and 'test' attributes to node with set probability.
  '''
  global currentNodeId
  if paperId in paperId2NodeId:
    nodeId = paperId2NodeId[paperId]
  else:
    nodeId = currentNodeId
    paperId2NodeId[paperId] = nodeId
    currentNodeId += 1
  if not G.has_node(nodeId):
    if enableAttributes:
      r = random.uniform(0, 1)
      if r <= pTest:
        G.add_node(nodeId, test=True, val=False)
        return
      elif r <= pVal:
        G.add_node(nodeId, test=False, val=True)
        return
    G.add_node(nodeId, test=False, val=False)
      

def appendEdge(G, paper1, paper2, label, ingestionFlags):
  global validationEdges
  shouldIngest = ingestionFlags.get(label, False)
  if shouldIngest:
    r = random.uniform(0, 1)
    node1 = paperId2NodeId[paper1]
    node2 = paperId2NodeId[paper2]
    if r <= pEdgeVal and G.degree(node1) > 1 and G.degree(node2) > 1:
      validationEdges.add((node1, node2))
      #G.add_edge(node1, node2, type=label)
    else:
      G.add_edge(node1, node2, type=label)


def shouldSample(paperId):
  return paperId % 100 > samplingRate


def loadGraph(fileName, ingestionFlags):
  G = nx.MultiGraph()
  authorMap = dict()  # key: author, value: list of paperId
  publicationMap = dict()  # key: publication+year, value: list of paperId
  paperFeaturesMap = dict()  # key: research paper, value: (year, title, abstract)
  blacklist = set()
  # In each iteration we keep the following information until we see the new line
  # authors: list of authors
  # references: list of references
  # currentPaperId
  # currentPaperTitle
  # currentPaperAbstract
  # publicationYear
  # publicationVenue
  (currentPaperTitle, currentPaperAbstract, authorsRaw, publicationYear, publicationVenue,
    currentPaperId) = ("", "", "", "", "", "")
  references = []
  count = 0
  with open(fileName, 'r') as f:
    for line in f.readlines():
      prefix = line[:2]
      if prefix == "#*":
        currentPaperTitle = line[2:].strip()
      elif prefix == "#!":
        currentPaperAbstract = line[2:].strip()
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
        appendNode(G, currentPaperId)
        paperFeaturesMap[currentPaperId] = (int(publicationYear), currentPaperTitle, currentPaperAbstract)
        for reference in references:
          if reference not in blacklist:
            appendNode(G, reference)
            appendEdge(G, currentPaperId, reference, "reference", ingestionFlags)
        if len(authorsRaw) >= 1:
          authors = authorsRaw.split(',')
          for author in authors:
            if author not in knownInvalidAuthorNames:
              authorMap.setdefault(author,[]).append(currentPaperId)
        hashKey = publicationVenue + publicationYear
        publicationMap.setdefault(hashKey,[]).append(currentPaperId)

        (currentPaperTitle, currentPaperAbstract, authorsRaw, publicationYear, publicationVenue,
         currentPaperId) = ("", "", "", "", "", "")
        references = []
  print "After adding reference edges:"
  printGraphStat(G)
  annotateGraphWithEdges(G, authorMap, "coauthor", ingestionFlags)
  print "After adding coauthor edges:"
  printGraphStat(G)
  #annotateGraphWithEdges(G, publicationMap, 'publication', ingestionFlags)
  #print "After adding publication edges:"
  #printGraphStat(G)
  return G


def annotateGraphWithEdges(G, map, t, ingestionFlags):
  '''
  map: a dictionary of <key, list[paperId]>
  For each key in the map, add pair-wise edge to every node denoted by value 
  '''
  for k in map:
    papers = map[k]
    for index1 in xrange(len(papers)):
      for index2 in xrange(index1+1, len(papers)):
        appendEdge(G, papers[index1], papers[index2], t, ingestionFlags)


def printGraphStat(G):
  print G.number_of_nodes(), G.number_of_edges()


def printNodeStat(G):
  numVal = 0
  numTest = 0
  numTrain = 0
  for node in G.__iter__():
    if "val" in G.node[node] and G.node[node]["val"]:
      numVal += 1
    elif "test" in G.node[node] and G.node[node]["test"]:
      numTest += 1
    else:
      numTrain += 1
  print("Number of nodes labeled as val, test, train are %d, %d, %d"
    % (numVal, numTest, numTrain))


def dumpAsJson(G, path_prefix, dumpFeatures):
  """Dumps the graph data as json.

  The format dumped is parsable by GraphSAGE util method.
  """
  data = json_graph.node_link_data(G)
  with open("{}-G.json".format(path_prefix), "w") as f:
    json.dump(data, f)
  with open("{}-id_map.json".format(path_prefix), "w") as f:
    node_name_id_map = {}
    idx = 0
    for node_id in sorted(G.nodes()):
      node_name_id_map[str(node_id)] = idx
      idx += 1
    json.dump(node_name_id_map, f)
  with open("{}-feats.npy".format(path_prefix), "w") as f:
    features = []
    numNodesWithoutFeatures = 0
    for node_id in sorted(G.nodes()):
      if "feature" in G.node[node_id]:
        features.append(G.node[node_id]["feature"])
      else:
        features.append([0])
        numNodesWithoutFeatures += 1
    print("Number of nodes without features: {}".format(numNodesWithoutFeatures))
    if dumpFeatures:
      np.save(f, features)
  with open("{}-class_map.json".format(path_prefix), "w") as f:
    node_name_class_map = {}
    for node_id in sorted(G.nodes()):
      # Load dummy class data
      node_name_class_map[str(node_id)] = [1]
    json.dump(node_name_class_map, f)
  with open("{}-walks.txt".format(path_prefix), "w") as f:
    f.write("0\t1")  # Write dummy data


def selectValidationNonEdges(G):
  validationNonEdges = set()
  n = G.number_of_nodes()
  size = 0
  target = len(validationEdges)
  while size < target:
    randomPaper1 = random.randint(0, n)
    randomPaper2 = random.randint(0, n)
    if not G.has_edge(randomPaper1, randomPaper2):
      validationNonEdges.add((randomPaper1, randomPaper2))
      size += 1
  return validationNonEdges


ingestionFlags = {
  "reference": True,
  "coauthor": False,
  "publication": False
}

G = loadGraph("outputacm.txt", ingestionFlags)
validationNonEdges = selectValidationNonEdges(G)

with open('validation_edges', 'wb') as file:
  pickle.dump(validationEdges, file)
with open('validation_nonedges', 'wb') as file:
  pickle.dump(validationNonEdges, file)
with open('graph_stat', 'wb') as file:
  pickle.dump((G.number_of_nodes(), G.number_of_edges()), file)

printNodeStat(G)
dumpAsJson(G, "acm", True)
