import unittest
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np

# Afegir el directori pare 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import reader as rd
import data_paths as dp
from graph import ELDP

class TestELDP(unittest.TestCase):
    __slots__ = ('dictionary_options', 'readers', 'ELDP', 'save', 'numExperiments')
    def setUp(self):
        """Crea una instància de ELDP
        """
        # En cas de testejar !
        # self.save = False # Canviar si no es volen guardar els grafs resultants
        # self.dictionary_options = {'1': (dp.DATASET1, True, False, 'FILE'), 
        #                 '2': (dp.DATASET3, True, True, 'FILE')} 
        # grouping = None     
        # epsilons = [1, 10]
        # self.numExperiments = 1 

        self.save = True # Canviar si no es volen guardar els grafs resultants
        self.dictionary_options = {'1': (dp.DATASET3, True, False, 'FILE')} 
        self.numExperiments = 5     
        grouping = None
        epsilons = [0.1, 2, 4, 6, 8, 10, 20]     

        self.readers = [] # LLegim els fitxers, i els guardem en una llista 
        for key, value in self.dictionary_options.items():
            self.readers.append(rd.Reader(value))
        
        self.ELDP = []
        for eps in epsilons: # Per cada epsilon, creem una instància de ELDP amb tots els fitxers
            for i,reader in enumerate(self.readers):
                self.ELDP.append(ELDP(reader.filename, self.dictionary_options[str(i+1)], reader.df, grouping, eps))

        # nx.draw(self.ELDP[0].graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=16)
        # plt.show()

    def test_gilbert_graph(self):
        """Testing noise-graphs...
        """
        # 1. Comprovar que els nodes del noise-graph són els mateixos
        for g in self.ELDP:
            G = g.gilbert_graph(0.5)
            
            # Comprovar que són dirigits o no segons el graf original
            if g.directed:
                self.assertIsInstance(G, nx.DiGraph)
            else:
                self.assertIsInstance(G, nx.Graph)

            self.assertEqual(G.number_of_nodes(), g.graph.number_of_nodes())
            self.assertEqual(list(g.graph.nodes()), list(G.nodes()))

        # print(f"Nodes Gilbert noise graph: OK")

    def test_protection(self):
        """Testing ELDP protection...
        """
        for e in range(self.numExperiments):
            for i,g in enumerate(self.ELDP):
                
                # print(f"Dataset: {self.ELDP[i].filename}, Epsilon: {self.ELDP[i].epsilon}")

                original_g, protected_g = g.apply_protection()
            
                # 1. Comprovar densitat // ε-ELDP
                for og, pr in zip(original_g, protected_g):
                    density_og = nx.density(og)
                    density_pr = nx.density(pr)
                    self.assertAlmostEqual(density_og, density_pr, delta=0.05)
                    # print(f"Densitat original: {density_og}, Densitat protegit: {density_pr}")

                    p0,p1 = self.ELDP[i].compute_probabilities(density_og)
                    constraint = round(math.exp(self.ELDP[i].epsilon),3)
                    f1  = (1-p1) / p0
                    f2 = p1 / (1-p0)
                    f3 = p0 / (1-p1) 
                    f4 = (1-p0) / p1
                    maxim = round(max(f1,f2,f3,f4),3)
                    self.assertGreaterEqual(constraint, maxim)
                    # print(f"Compleix ε-ELDP: {constraint} >= {maxim}")
                    # print("================================================")

                
                # 2. Save/Load grafs
                if self.save: 
                    g.save_graphs(original_g, protected_g, "ELDP", e+1, self.ELDP[i].epsilon)
            

    # def test_saved_graphs(self):
    #     """5. Test loading saved graphs
    #     """
    #     with open("code/output/ELDP/LNetwork.json_2.pkl", "rb") as f:
    #         graphs = pickle.load(f)
    #     self.assertIsInstance(graphs, list)
    #     self.assertIsInstance(graphs[0], nx.Graph)

    #     with open("code/output/ELDP/LNetwork.json_3.pkl", "rb") as f:
    #         graphs = pickle.load(f)
    #     self.assertIsInstance(graphs, list)
    #     self.assertIsInstance(graphs[0], nx.Graph)    

    #     with open("code/output/ELDP/LNetwork.json_4.pkl", "rb") as f:
    #         graphs = pickle.load(f)
    #     self.assertIsInstance(graphs, list)
    #     self.assertIsInstance(graphs[0], nx.Graph)

    # def test_density(self):
    #     """1. Test de densitat
    #     """
    #     # 1. Comprovació per grafs no dirigits
    #     self.ELDP[0].graph = nx.Graph()
    #     self.ELDP[0].graph.add_nodes_from([1, 2, 3, 4])
    #     self.ELDP[0].graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
    #     manual_density_undirected = (2*3)/(4*3)
    #     self.assertEqual(manual_density_undirected, nx.density(self.ELDP[0].graph))
    #     # print(f"Density for undirected: OK")

    #     # 2. Comprovació per grafs dirigits
    #     self.ELDP[0].graph = nx.DiGraph()
    #     self.ELDP[0].graph.add_nodes_from([1, 2, 3, 4])
    #     self.ELDP[0].graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
    #     manual_density_directed = (3)/(4*3)
    #     self.assertEqual(manual_density_directed, nx.density(self.ELDP[0].graph))  
    #     # print(f"Density for directed: OK")

    #     # 3. Còmput de les densitats individualment
    #     for _, group in self.ELDP[1].grouped_df:
    #         self.ELDP[1].iterate_graph(group)
    #         d = nx.density(self.ELDP[1].graph)
    #         self.assertGreater(d,0)
    #         self.assertLessEqual(d,0.5)
    #     # print(f"Density individually: OK")

    # def test_complement_graph(self):
    #     """2. Test graf complementari
    #     """
    #     # 1. Per un graf no dirigit
    #     self.ELDP[0].graph = nx.Graph()
    #     self.ELDP[0].graph.add_nodes_from([1, 2, 3, 4])
    #     self.ELDP[0].graph.add_edges_from([(1, 2), (2, 3), (3, 4)])

    #     ELDP2 = ELDP(self.readers[0].filename, self.dictionary_options["1"], self.readers[0].df, 0.5)
    #     ELDP2.graph = nx.Graph()
    #     ELDP2.graph.add_nodes_from([1, 2, 3, 4])
    #     ELDP2.graph.add_edges_from([(4, 2), (4, 1), (3, 1)])
        
    #     complement = nx.complement(self.ELDP[0].graph)

    #     correct_result = set([(2, 4), (1, 3), (1, 4)])
    #     actual_result = set(list(ELDP2.graph.edges()) + list(complement.edges()))
    #     self.assertEqual(actual_result, correct_result)
    #     # print(f"Complement graph for undirected: OK")

    #     # 2. Per un graf dirigit    
    #     self.ELDP[0].graph = nx.DiGraph()
    #     self.ELDP[0].graph.add_nodes_from([1, 2, 3, 4])
    #     self.ELDP[0].graph.add_edges_from([(1, 2), (2, 3), (3, 4)])

    #     ELDP2 = ELDP(self.readers[0].filename, self.dictionary_options["1"], self.readers[0].df, 0.5)
    #     ELDP2.graph = nx.DiGraph()
    #     ELDP2.graph.add_nodes_from([1, 2, 3, 4])
    #     ELDP2.graph.add_edges_from([(1, 3), (1, 4), (2, 1), (2, 4), 
    #                                  (3, 2), (3, 1), (4, 3), (4, 2), (4, 1)])
        
    #     complement = nx.complement(self.ELDP[0].graph)
    #     correct_result = set([(1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)])
    #     actual_result = set(list(ELDP2.graph.edges()) + list(complement.edges()))
    #     self.assertEqual(actual_result, correct_result)
    #     # print(f"Complement for directed: OK")

    # def test_operations_graph(self):
    #     """4. Test intersection and XOR
    #     """
    #     # 1. Comprovar que realment fa l'intersecció com es vol
    #     g1 = nx.Graph()
    #     g1.add_nodes_from([1, 2, 3, 4])
    #     g1.add_edges_from([(1, 2), (2, 3), (3, 4)])
    #     g2 = nx.Graph()
    #     g2.add_nodes_from([1, 2, 3, 4])
    #     g2.add_edges_from([(1, 2), (1, 3), (1, 4)])

    #     correct_result = [(1,2)]
    #     actual_result = nx.intersection(g1, g2)
    #     self.assertEqual(list(actual_result.edges()), correct_result)

    #     # 2. Comprovar XOR
    #     g1 = nx.Graph()
    #     g1.add_nodes_from([1, 2, 3, 4])
    #     g1.add_edges_from([(1, 2), (2, 3), (3, 4)])
    #     g2 = nx.Graph()
    #     g2.add_nodes_from([1, 2, 3, 4])
    #     g2.add_edges_from([(1, 3), (1, 2), (2, 4)])

    #     correct_result = [(1, 3), (2, 3), (2, 4), (3, 4)]
    #     actual_result = nx.symmetric_difference(g1, g2)
    #     self.assertEqual(list(actual_result.edges()), correct_result)
    
    # def test_epsilon(self):
    #     """6. Test per veure com funciona l'algorisme canviant el paràmetre epsilon
    #     """
    #     epsilons = np.arange(0.01, 2, 0.05)  # Desde 0.1 hasta 2.0 con paso de 0.1
    #     eps_grafs = [ELDP(self.readers[2].filename, self.dictionary_options['3'], self.readers[2].df, e) for e in epsilons]
        
    #     p00_values = [obj.p0 for obj in eps_grafs]
    #     p01_values = [1 - obj.p0 for obj in eps_grafs]
    #     p11_values = [obj.p1 for obj in eps_grafs]
    #     p10_values = [1 - obj.p1 for obj in eps_grafs]


    #     # Graficar p0 i p1 values
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(epsilons, p00_values, marker='o', linestyle='-', label=r'$p00$')
    #     plt.plot(epsilons, p01_values, marker='o', linestyle='-', label=r'$p01$')
    #     plt.plot(epsilons, p11_values, marker='s', linestyle='-', label=r'$p11$')
    #     plt.plot(epsilons, p10_values, marker='s', linestyle='-', label=r'$p10$')

    #     # Etiquetas y título
    #     plt.xlabel(r'$\epsilon$', fontsize=12)
    #     plt.ylabel('Probabilitat', fontsize=12)
    #     plt.title('Variació de probabilitats segons ε', fontsize=14)
    #     plt.legend()
    #     plt.grid(True)

    #     # Mostrar el gráfico
    #     plt.show()

if __name__ == '__main__':
    unittest.main()
