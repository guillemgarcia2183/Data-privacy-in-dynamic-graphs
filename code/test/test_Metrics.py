import unittest
import sys
import os
import networkx as nx

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
        self.g3.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 3)])

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
        matrix, maxdegree = self.metrics.degreeMatrices(self.g1, None)
        self.assertEqual(matrix.tolist(), [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])
        self.assertEqual(maxdegree, 2)

        matrix, maxdegree = self.metrics.degreeMatrices(self.g3, "in")
        self.assertEqual(matrix.tolist(), [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])
        self.assertEqual(maxdegree, 2)

        # Score matrices
        
        

if __name__ == '__main__':
    unittest.main()
