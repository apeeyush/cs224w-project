import pickle
import numpy as np
import random

random.seed(0)
data_dir = "GraphSAGE/unsup-../graphsage_mean_small_0.000010"
embeds = np.load(data_dir + "/val.npy")
id_map = {}
with open(data_dir + "/val.txt") as fp:
  for i, line in enumerate(fp):
    id_map[int(line.strip())] = i


def getNodeEmbedding(nodeId):
  return embeds[id_map[nodeId]]


def loadValidationEdges():
  with open ('validation_edges', 'rb') as file:
    validationEdges = pickle.load(file)
  with open ('validation_nonedges', 'rb') as file:
    validationNonEdges = pickle.load(file)
  return validationEdges, validationNonEdges


def loadGraphStat():
  with open ('graph_stat', 'rb') as file:
    (numNode, numEdge) = pickle.load(file)
  return numNode, numEdge


def predictEdgeBasedOnEmbedding(node1, node2):
  '''
  Given two nodes, predict the likelihood of an edge existing between them
  For now, this is the l2 distance between the embeddings of the two nodes
  '''
  embedding1 = getNodeEmbedding(node1)
  embedding2 = getNodeEmbedding(node2)
  return np.linalg.norm(embedding1 - embedding2, 2)


def coinToss(r):
  toss = random.uniform(0, 1)
  return toss < r


def evaluateAUC():
  validationEdges, validationNonEdges = loadValidationEdges()
  numNode, numEdge = loadGraphStat()
  edgeDensity = numEdge / (numNode * (numNode-1) / 2.0)
  edgeWin, nonEdgeWin = 0,0
  for edge, nonEdge in zip(validationEdges, validationNonEdges):
    edgeDistance = predictEdgeBasedOnEmbedding(edge[0], edge[1])
    nonEdgeDistance = predictEdgeBasedOnEmbedding(nonEdge[0], nonEdge[1])
    if edgeDistance < nonEdgeDistance:
      edgeWin += 1
    else:
      nonEdgeWin += 1
  print("Edge density of the graph is %f" % (edgeDensity))
  print("There are a total of %d validation edges and %d validation non-edges" % (len(validationEdges), len(validationNonEdges)))
  print("Number of times a random validation edge is more similar: %d" % (edgeWin))
  print("Number of times a random validation non-edge is more similar: %d" % (nonEdgeWin))


def evaluate():
  validationEdges, validationNonEdges = loadValidationEdges()
  numNode, numEdge = loadGraphStat()
  edgeDensity = numEdge / (numNode * (numNode-1) / 2.0)
  edgeProbEmbed, nonEdgeProbEmbed = 0,0
  for edge in validationEdges:
    edgeProbEmbed += predictEdgeBasedOnEmbedding(edge[0], edge[1])
  for edge in validationNonEdges:
    nonEdgeProbEmbed += predictEdgeBasedOnEmbedding(edge[0], edge[1])
  edgeProbEmbed /= len(validationEdges)
  nonEdgeProbEmbed /= len(validationNonEdges)
  print("Edge density of the graph is %f" % (edgeDensity))
  print("There are a total of %d validation edges and %d validation non-edges" % (len(validationEdges), len(validationNonEdges)))
  print("Among all validation edges, average l2 distance of a predicted link is %f" % (edgeProbEmbed))
  print("Among all validation non-edges, average l2 distance of a predicted link is %f" % (nonEdgeProbEmbed))


evaluateAUC()
