import unittest
import sys
import os
import numpy as np

# Afegir el directori pare 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import reader as rd
import data_paths as dp
from graph import KDA

class TestKDA(unittest.TestCase):
    __slots__ = ('dictionary_options', 'readers', 'KDA')
    def setUp(self):
        """Crea una instància de KDA
        """
        self.dictionary_options = {'1': (dp.DATASET1, 'weighted', 'undirected'), 
                        '2': (dp.DATASET2, 'weighted', 'undirected')}
        
        self.readers = []
        for key, value in self.dictionary_options.items():
            self.readers.append(rd.Reader(value))
        
        self.KDA = []
        for i,reader in enumerate(self.readers):
            self.KDA.append(KDA(reader.filename, self.dictionary_options[str(i+1)], reader.df))

        # nx.draw(self.ELDP[0].graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=16)
        # plt.show()

    def test_degrees(self):
        """1. Test degree matrices
        """
        for g in self.KDA:
            if g.directed == "directed":
                self.assertEqual(g.indegree_matrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.indegree_matrix[0].dtype, 'int32')
                self.assertEqual(g.outdegree_matrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.outdegree_matrix[0].dtype, 'int32')
                self.assertEqual(g.degree_matrix, None)
            else:
                self.assertEqual(g.degree_matrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.degree_matrix[0].dtype, 'int32')
                self.assertEqual(g.indegree_matrix, None)
                self.assertEqual(g.outdegree_matrix, None)

    def test_P_matrix(self):
        pass

if __name__ == '__main__':
    unittest.main()
