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
        self.metrics = Metrics(list(), list())


    def test_jaccard(self):
        """Testing coeficient de Jaccard
        """
        g1 = nx.Graph()
        g1.add_edges_from([(1, 2), (2, 3), (3, 2), (4, 3), (3, 4)])

        g2 = nx.Graph()
        g2.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 4)])

        g3 = nx.DiGraph()
        g3.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 3)])

        g4 = nx.DiGraph()
        g4.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 4)])

        g5 = nx.Graph()
               
        r1 = self.metrics.jaccardCoefficient(g1, g2)
        self.assertEqual(r1, (2/4)*100)
        r2 = self.metrics.jaccardCoefficient(g3, g4)
        self.assertEqual(r2, (2/6)*100)
        r3 = self.metrics.jaccardCoefficient(g1, g5)
        self.assertEqual(r3, (0.0)*100)

if __name__ == '__main__':
    unittest.main()
