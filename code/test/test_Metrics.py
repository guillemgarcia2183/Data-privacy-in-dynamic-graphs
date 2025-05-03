import unittest
import sys
import os
import networkx as nx
import numpy as np

# Afegir el directori pare 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from metrics import Metrics

class TestMetrics(unittest.TestCase):
    __slots__ = ('dictionary_options', 'readers', 'KDA', 'save')
    def setUp(self):
        """Crea una instància de KDA
        """
        self.metrics = Metrics()
        
        # Grafs de prova
        self.g1 = nx.Graph()
        self.g1.add_edges_from([(1, 2), (2, 3), (3, 4)])

        self.g2 = nx.Graph()
        self.g2.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 4)])

        self.g3 = nx.DiGraph()
        self.g3.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 3)])

        self.g4 = nx.DiGraph()
        self.g4.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 4)])

        self.g5 = nx.Graph()

    def test_jaccard(self):
        """Testing coeficient de Jaccard
        """
        r1 = self.metrics.jaccardIndex(self.g1, self.g2)
        self.assertEqual(r1, (2/4)*100)
        r2 = self.metrics.jaccardIndex(self.g3, self.g4)
        self.assertEqual(r2, (2/6)*100)
        r3 = self.metrics.jaccardIndex(self.g1, self.g5)
        self.assertEqual(r3, (0.0)*100)
    
    def test_DeltaCon(self):
        """Testing DeltaCon: matrius de graus, scores, etc.
        """
        # Matriu de graus
        degreeMatrix, maxDegree = self.metrics.degreeMatrices(self.g1)
        self.assertEqual(degreeMatrix.tolist(), [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])
        self.assertEqual(maxDegree, 2)

        # Score matrices - Adjacència, identitat i influència
        identityMatrix = np.identity(self.g1.number_of_nodes())
        self.assertEqual(identityMatrix.shape, (4, 4))
        adjacencyMatrix = nx.adjacency_matrix(self.g1).toarray()
        self.assertEqual(adjacencyMatrix.shape, (4, 4))
        influence = self.metrics.influenceNeighbors(maxDegree)
        squaredInfluence = influence * influence
        self.assertEqual(influence, 1/3)
        self.assertEqual(squaredInfluence, 1/9)
        S = identityMatrix + (squaredInfluence * degreeMatrix) - (influence * adjacencyMatrix)
        self.assertEqual(S.shape, (4, 4))
        finalMatrix = np.linalg.inv(S)

        scoreMatrix = self.metrics.scoreMatrix(self.g1)
        self.assertEqual(scoreMatrix.tolist(), finalMatrix.tolist())

        # Root Euclidean distance
        S1 = self.metrics.scoreMatrix(self.g1)
        S2 = self.metrics.scoreMatrix(self.g1)
        rootED = self.metrics.rootEuclideanDistance(S1, S2)
        self.assertEqual(rootED, 0.0)

        S11 = self.metrics.scoreMatrix(self.g3)
        S22 = self.metrics.scoreMatrix(self.g4)
        rootED = self.metrics.rootEuclideanDistance(S11, S22)
        self.assertGreater(rootED, 0)

    def test_centralities(self):
        """Testing densitats i centralitats
        """
        densityOriginal, densityProtected = self.metrics.getDensities(self.g1, self.g2)
        self.assertGreaterEqual(densityOriginal, 0)
        self.assertGreaterEqual(densityProtected, 0)
        self.assertLessEqual(densityOriginal, 1)
        self.assertLessEqual(densityProtected, 1)
        
        c1 = nx.betweenness_centrality(self.g1)
        c2 = nx.closeness_centrality(self.g1)
        c3 = nx.degree_centrality(self.g1)
        self.assertIsInstance(c1, dict)
        self.assertIsInstance(c2, dict)
        self.assertIsInstance(c3, dict)

if __name__ == '__main__':
    unittest.main()
