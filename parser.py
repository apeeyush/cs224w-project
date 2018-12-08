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
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from networkx.readwrite import json_graph
import pickle
import gensim

random.seed(0)

knownInvalidAuthorNames = set()
knownInvalidAuthorNames.add(" II")
knownInvalidAuthorNames.add(" III")
knownInvalidAuthorNames.add(" IV")
knownInvalidAuthorNames.add(" V")
knownInvalidAuthorNames.add(" VI")
knownInvalidAuthorNames.add(" Jr.")
knownInvalidAuthorNames.add("Staff")

class DocEmbeddings(object):
  """Get document embedding using the pre-trained Google news word embeddings.

  TODO(apeeyush):
  * Implement SIF based weighted averaging of word2vec.
  * Tune the embeddings for the research paper dataset.
  """

  def __init__(self, **kwargs):
    nltk.download('punkt')
    self.model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    self.word_vectors = self.model.wv

  def get_word_embedding(self, word):
    try:
      embedding = self.model.word_vec(word)
      return True, embedding
    except KeyError:
      return False, []

  def get_sentence_embedding(self, sentence, debug=False):
    words = word_tokenize(sentence.lower().decode("utf8"))
    embeddings = []
    for word in words:
      present, embedding = self.get_word_embedding(word)
      if present:
        embeddings.append(embedding)
      else:
        # TODO(apeeyush): Consider splitting by '-' as well.
        if debug:
          print(word)
    if len(embeddings) > 0:
      return np.sum(embeddings, axis=0) / len(embeddings)
    else:
      return np.zeros(300)


pTest = 0.1  # [0, 0.1] falls into test set
pVal = 0  # no validation node set
enableAttributes = True

# Only ingest x% of nodes
samplingRate = 100

paperId2NodeId = dict()
currentNodeId = 0

pEdgeVal = 0.01 # 1% of edges are hidden from generated graph as validation edges
validationEdges = set()


WALK_LEN=5
N_WALKS=25
def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 10000 == 0:
            print("Done walks for", count, "nodes")
    return pairs


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
      elif prefix == "\n" or prefix == "\r\n":
        appendNode(G, currentPaperId)
        paperFeaturesMap[currentPaperId] = (int(publicationYear), currentPaperTitle, currentPaperAbstract)
        for reference in references:
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
  return G, paperFeaturesMap


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


def addNodeFeatures(G, nodeFeaturesMap):
  doc_embeddings = DocEmbeddings()
  nodeFeatureAttr = defaultdict(list)
  nodePublicationAttr = defaultdict(int)
  maxPublicationYear = max(v[0] for v in nodeFeaturesMap.values())
  minPublicationYear = min(v[0] for v in nodeFeaturesMap.values())
  for paperId, features in nodeFeaturesMap.items():
    if paperId in paperId2NodeId:
      nodeId = paperId2NodeId[paperId]
      nodePublicationAttr[nodeId] = features[0]
      nodeFeatureAttr[nodeId] = [0]
      # Add normalized publication year as feature
      if "publicationYear" in featureSet:
        normalizedPublicationYear = (features[0] - minPublicationYear) * 1.0 / maxPublicationYear
        nodeFeatureAttr[nodeId].append(normalizedPublicationYear)
      # Compute the feature for paper title
      if "paperTitle" in featureSet:
        titleEmbedding = doc_embeddings.get_sentence_embedding(features[1])
        nodeFeatureAttr[nodeId] += titleEmbedding
      # Compute the feature for paper abstract
      if "paperAbstract" in featureSet:
        abstractEmbedding = doc_embeddings.get_sentence_embedding(features[2])
        nodeFeatureAttr[nodeId] += abstractEmbedding
  nx.set_node_attributes(G, "feature", nodeFeatureAttr)
  nx.set_node_attributes(G, "publicationYear", nodePublicationAttr)


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
  with open("{}-id_map.json".format(path_prefix), "w") as f:
    node_name_id_map = {}
    idx = 0
    for node_id in sorted(G.nodes()):
      node_name_id_map[str(node_id)] = idx
      idx += 1
    json.dump(node_name_id_map, f)
  if dumpFeatures:
    with open("{}-feats.npy".format(path_prefix), "w") as f:
      features = []
      numNodesWithoutFeatures = 0
      for node_id in sorted(G.nodes()):
        if "feature" in G.node[node_id]:
          features.append(G.node[node_id]["feature"])
          del G.node[node_id]["feature"]
        else:
          features.append(np.zeros(featureSetSize))
          numNodesWithoutFeatures += 1
      print("Number of nodes without features: {}".format(numNodesWithoutFeatures))
      np.save(f, features)
  with open("{}-G.json".format(path_prefix), "w") as f:
    data = json_graph.node_link_data(G)
    json.dump(data, f)
  with open("{}-class_map.json".format(path_prefix), "w") as f:
    node_name_class_map = {}
    for node_id in sorted(G.nodes()):
      # Load dummy class data
      node_name_class_map[str(node_id)] = [1]
    json.dump(node_name_class_map, f)
  with open("{}-walks.txt".format(path_prefix), "w") as f:
    random_walks = run_random_walks(G, G.nodes())
    for (node1, node2) in random_walks:
      f.write("{}\t{}\n".format(node1, node2))


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
dumpFeatures = False
# "publicationYear", "paperTitle", "paperAbstract"
# Size: 1, 300, 300
featureSet = ["paperTitle", "paperAbstract"]
featureSetSize = 600

graph_filepath = "outputacm.txt"  # One of outputacm.txt or citation-network2.txt
G, paperFeaturesMap = loadGraph(graph_filepath, ingestionFlags)
validationNonEdges = selectValidationNonEdges(G)
if dumpFeatures:
  addNodeFeatures(G, paperFeaturesMap)

with open('validation_edges', 'wb') as file:
  pickle.dump(validationEdges, file)
with open('validation_nonedges', 'wb') as file:
  pickle.dump(validationNonEdges, file)
with open('graph_stat', 'wb') as file:
  pickle.dump((G.number_of_nodes(), G.number_of_edges()), file)

printNodeStat(G)
dumpAsJson(G, "acm", dumpFeatures)
