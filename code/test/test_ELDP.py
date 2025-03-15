import unittest
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import math
import pickle

# Afegir el directori pare 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import reader as rd
import data_paths as dp
from graph import ELDP

class TestELDP(unittest.TestCase):
    __slots__ = ('dictionary_options', 'readers', 'ELDP')
    def setUp(self):
        """Crea una instància de ELDP
        """
        self.dictionary_options = {'1': (dp.DATASET1, 'weighted', 'undirected'), 
                        '2': (dp.DATASET2, 'weighted', 'undirected'),
                        '3': (dp.DATASET3, 'weighted', 'undirected')}
        
        self.readers = []
        for key, value in self.dictionary_options.items():
            self.readers.append(rd.Reader(value))
        
        self.ELDP = []
        for i,reader in enumerate(self.readers):
            self.ELDP.append(ELDP(reader.filename, self.dictionary_options[str(i+1)], reader.df))

        # nx.draw(self.ELDP[0].graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=16)
        # plt.show()

    def test_density(self):
        """1. Test de densitat
        """
        # 1. Comprovació per grafs no dirigits
        self.ELDP[0].graph = nx.Graph()
        self.ELDP[0].graph.add_nodes_from([1, 2, 3, 4])
        self.ELDP[0].graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
        manual_density_undirected = (2*3)/(4*3)
        self.assertEqual(manual_density_undirected, nx.density(self.ELDP[0].graph))
        # print(f"Density for undirected: OK")

        # 2. Comprovació per grafs dirigits
        self.ELDP[0].graph = nx.DiGraph()
        self.ELDP[0].graph.add_nodes_from([1, 2, 3, 4])
        self.ELDP[0].graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
        manual_density_directed = (3)/(4*3)
        self.assertEqual(manual_density_directed, nx.density(self.ELDP[0].graph))  
        # print(f"Density for directed: OK")

        # 3. Comprovació densitat global <= 0.5 -- Necessari per poder assegurar de preservar densitat i complir epsilon LEDP
        for graph in self.ELDP:
            self.assertLessEqual(graph.density, 0.5)
        # print(f"Density <= 0.5: OK")

        # 4. Còmput de les densitats individualment
        list_densities = []
        for _, group in self.ELDP[2].grouped_df:
            self.ELDP[2].iterate_graph(group)
            list_densities.append(nx.density(self.ELDP[2].graph))
        for d in list_densities:
            self.assertGreater(d,0)
            self.assertLessEqual(d,1)
        # print(f"Density individually: OK")

    def test_complement_graph(self):
        """2. Test graf complementari
        """
        # 1. Per un graf no dirigit
        self.ELDP[0].graph = nx.Graph()
        self.ELDP[0].graph.add_nodes_from([1, 2, 3, 4])
        self.ELDP[0].graph.add_edges_from([(1, 2), (2, 3), (3, 4)])

        ELDP2 = ELDP(self.readers[0].filename, self.dictionary_options["1"], self.readers[0].df)
        ELDP2.graph = nx.Graph()
        ELDP2.graph.add_nodes_from([1, 2, 3, 4])
        ELDP2.graph.add_edges_from([(4, 2), (4, 1), (3, 1)])
        
        complement = nx.complement(self.ELDP[0].graph)

        correct_result = set([(2, 4), (1, 3), (1, 4)])
        actual_result = set(list(ELDP2.graph.edges()) + list(complement.edges()))
        self.assertEqual(actual_result, correct_result)
        # print(f"Complement graph for undirected: OK")

        # 2. Per un graf dirigit    
        self.ELDP[0].graph = nx.DiGraph()
        self.ELDP[0].graph.add_nodes_from([1, 2, 3, 4])
        self.ELDP[0].graph.add_edges_from([(1, 2), (2, 3), (3, 4)])

        ELDP2 = ELDP(self.readers[0].filename, self.dictionary_options["1"], self.readers[0].df)
        ELDP2.graph = nx.DiGraph()
        ELDP2.graph.add_nodes_from([1, 2, 3, 4])
        ELDP2.graph.add_edges_from([(1, 3), (1, 4), (2, 1), (2, 4), 
                                     (3, 2), (3, 1), (4, 3), (4, 2), (4, 1)])
        
        complement = nx.complement(self.ELDP[0].graph)
        correct_result = set([(1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)])
        actual_result = set(list(ELDP2.graph.edges()) + list(complement.edges()))
        self.assertEqual(actual_result, correct_result)
        # print(f"Complement for directed: OK")

    def test_gilbert_graph(self):
        """3. Test noise-graphs
        """
        # 1. Comprovar que els nodes del noise-graph són els mateixos
        for g in self.ELDP:
            G = nx.erdos_renyi_graph(g.graph.number_of_nodes(), 0.5)
            self.assertEqual(list(g.graph.nodes()), list(G.nodes()))
        # print(f"Nodes Gilbert noise graph: OK")

    def test_operations_graph(self):
        """4. Test intersection and XOR
        """
        # 1. Comprovar que realment fa l'intersecció com es vol
        g1 = nx.Graph()
        g1.add_nodes_from([1, 2, 3, 4])
        g1.add_edges_from([(1, 2), (2, 3), (3, 4)])
        g2 = nx.Graph()
        g2.add_nodes_from([1, 2, 3, 4])
        g2.add_edges_from([(1, 2), (1, 3), (1, 4)])

        correct_result = [(1,2)]
        actual_result = nx.intersection(g1, g2)
        self.assertEqual(list(actual_result.edges()), correct_result)

        # 2. Comprovar XOR
        g1 = nx.Graph()
        g1.add_nodes_from([1, 2, 3, 4])
        g1.add_edges_from([(1, 2), (2, 3), (3, 4)])
        g2 = nx.Graph()
        g2.add_nodes_from([1, 2, 3, 4])
        g2.add_edges_from([(1, 3), (1, 2), (2, 4)])

        correct_result = [(1, 3), (2, 3), (2, 4), (3, 4)]
        actual_result = nx.symmetric_difference(g1, g2)
        self.assertEqual(list(actual_result.edges()), correct_result)

    def test_protection(self):
        """Test dataset l-EDP protection
        """
        for i,g in enumerate(self.ELDP):
            original_g, protected_g = g.apply_protection()
        
            # 1. Comprovar densitats de grafs protegits
            density_protected = 0
            n = 0
            for j,graph in enumerate(protected_g):
                density_protected += nx.density(graph)
                n += 1
                self.assertNotEqual(list(graph.edges()), list(original_g[j].edges()))
            density_protected = density_protected/n
            
            #print(f"Densitat original DATASET [{i+1}]: {g.density} \n Densitat PROTEGIT: {density_protected} \n")
            self.assertAlmostEqual(density_protected, g.density, delta=0.1)
            
            # 2. Comprovar ε-ELDP
            epsilon = 0.5
            constraint = round(math.exp(epsilon), 8)
            f1  = (1-g.p1) / g.p0 
            f2 = g.p1 / (1-g.p0)
            f3 = g.p0 / (1-g.p1) 
            f4 = (1-g.p0) / g.p1
            maxim = round(max(f1,f2,f3,f4), 8)
            self.assertGreaterEqual(constraint, maxim)

            # 3. Save/Load grafs 
            g.save_graphs(original_g, protected_g)
        
        with open("code/output/ELDP/protected_graphs_aves-sparrow-social.edges.pkl", "rb") as f:
            protected_graphs = pickle.load(f)
        self.assertIsInstance(protected_graphs, list)
        self.assertIsInstance(protected_graphs[0], nx.Graph)
            
        # Densitat original DATASET [1]: 0.1902302285210165 
        #  Densitat PROTEGIT: 0.1897677793904209 

        # Densitat original DATASET [2]: 6.843660815597166e-05 
        #  Densitat PROTEGIT: 6.73125278786053e-05 

        # Densitat original DATASET [3]: 0.4075884953895978 
        #  Densitat PROTEGIT: 0.40899804508881005 

if __name__ == '__main__':
    unittest.main()
