import unittest
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt

# Afegir el directori pare 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import reader as rd
import data_paths as dp
from graph import LEDP

class TestLEDP(unittest.TestCase):
    __slots__ = ('L_EDP', 'L_EDP_DF')
    def setUp(self):
        """Crea una instància de Dataset
        """
        TUPLE1 = (dp.DATASET1, 'weighted', 'undirected')        
        TUPLE2 = (dp.DATASET3, 'weighted', 'undirected')
        reader1 = rd.Reader(TUPLE1)
        reader2 = rd.Reader(TUPLE2)
        self.L_EDP = LEDP(reader1.filename, TUPLE1, reader1.df)
        self.L_EDP_DF = LEDP(reader2.filename, TUPLE2, reader2.df)
        # nx.draw(self.L_EDP.graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=16)
        # plt.show()

    def test_density(self):
        """1. Test de densitat
        """
        # 1. Comprovació per grafs no dirigits
        self.L_EDP.graph = nx.Graph()
        self.L_EDP.graph.add_nodes_from([1, 2, 3, 4])
        self.L_EDP.graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
        manual_density_undirected = (2*3)/(4*3)
        self.assertEqual(manual_density_undirected, nx.density(self.L_EDP.graph))
        
        # 2. Comprovació per grafs dirigits
        self.L_EDP.graph = nx.DiGraph()
        self.L_EDP.graph.add_nodes_from([1, 2, 3, 4])
        self.L_EDP.graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
        manual_density_directed = (3)/(4*3)
        self.assertEqual(manual_density_directed, nx.density(self.L_EDP.graph))  

        # 3. Comprovació densitat global <= 0.5 -- Necessari per poder assegurar de preservar densitat i complir epsilon LEDP
        self.assertLessEqual(self.L_EDP_DF.density, 0.5)
        
        # 4. Veure individualment les densitats
        list_densities = []
        for _, group in self.L_EDP_DF.grouped_df:
            self.L_EDP_DF.iterate_graph(group)
            list_densities.append(nx.density(self.L_EDP_DF.graph))
        for d in list_densities:
            self.assertGreater(d,0)
            self.assertLessEqual(d,1)

if __name__ == '__main__':
    unittest.main()
