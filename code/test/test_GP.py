import unittest
import sys
import os
import networkx as nx

# Afegir el directori pare 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import reader as rd
import data_paths as dp
from graph import GraphProtection

class TestGP(unittest.TestCase):
    __slots__ = ('graph1', 'graph2')
    def setUp(self):
        """Crea una instància de Dataset
        """
        TUPLE1 = (dp.DATASET1, 'weighted', 'undirected', 'FILE')
        TUPLE2 = (dp.DATASET4, 'unweighted', 'directed', 'FILE')

        reader1 = rd.Reader(TUPLE1)
        reader2 = rd.Reader(TUPLE2)

        self.graph1 = GraphProtection(reader1.filename, TUPLE1, reader1.df)
        self.graph2 = GraphProtection(reader2.filename, TUPLE2, reader2.df)


    def test_graph_creation(self):
        """1. Test de creació de grafs
        """
        # 1. Comprovació per grafs no dirigits
        p1 = self.graph1.graph.is_directed()
        self.assertFalse(p1)

        # 2. Comprovació per grafs dirigits
        p2 = self.graph2.graph.is_directed()
        self.assertTrue(p2)
        
        # 3. Nombre de nodes
        n1 = self.graph1.graph.number_of_nodes()
        n2 = self.graph2.graph.number_of_nodes()
        self.assertGreater(n1, 0)
        self.assertGreater(n2, 0)

        # 4. Nombre d'arestes
        e1 = self.graph1.graph.number_of_edges()
        e2 = self.graph2.graph.number_of_edges()
        self.assertEqual(e1, 0)
        self.assertEqual(e2, 0)
    
    def test_iterations(self):
        """2. Test comprovar iteracions de grafs
        """
        list_edges = list()
        for _, group in self.graph1.grouped_df:
            self.graph1.iterate_graph(group)
            actual_edges = len(list(self.graph1.graph.edges()))
            list_edges.append(actual_edges)
        self.assertEqual(sum(list_edges), len(self.graph1.df))

    # def test_visualize_graph(self):
    #     self.graph2.visualize_graph()

if __name__ == '__main__':
    unittest.main()
