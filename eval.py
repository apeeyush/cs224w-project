import pickle
import numpy as np
import random

random.seed(0)

def getNodeEmbedding(nodeId):
  '''
  TODO:
  read the model trained by GraphSAGE, and return the node embedding as a vector given the nodeId
  '''
  return np.zeros(5)


def loadValidationEdges():
  with open ('validation_edges', 'rb') as file:
    validationEdges = pickle.load(file)
  with open ('validation_nonedges', 'rb') as file:
    validationNonEdges = pickle.load(file)
  return validationEdges, validationNonEdges


def loadGraphStat():
  with open ('graph_stat', 'rb') as file:
    (numEdge, numNode) = pickle.load(file)
  return numEdge, numNode


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


def evaluate():
  validationEdges, validationNonEdges = loadValidationEdges()
  numEdge, numNode = loadGraphStat()
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
  print("Among all validation edges, average probability of a predicted link is %f" % (edgeProbEmbed))
  print("Among all validation non-edges, average probability of a predicted link is %f" % (nonEdgeProbEmbed))


evaluate()
