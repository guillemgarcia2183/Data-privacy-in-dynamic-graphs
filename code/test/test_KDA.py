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
        self.dictionary_options = {'1': (dp.DATASET1, 'weighted', 'undirected', 'FILE'), 
                               '2': (dp.DATASET2, 'weighted', 'undirected', 'FILE'),
                               '3': (dp.DATASET3, 'weighted', 'undirected', 'FILE')}
        
        self.readers = []
        for key, value in self.dictionary_options.items():
            self.readers.append(rd.Reader(value))
        
        setK = np.arange(2, 15, 1)
        self.KDA = []
        for k in setK: 
            for i,reader in enumerate(self.readers):
                self.KDA.append(KDA(reader.filename, self.dictionary_options[str(i+1)], reader.df, k))

        # nx.draw(self.ELDP[0].graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=16)
        # plt.show()

    def test_degrees(self):
        """1. Test degree matrices
        """
        for g in self.KDA:
            if g.directed == "directed":
                self.assertEqual(g.indegreeMatrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.indegreeMatrix[0].dtype, 'int32')
                self.assertEqual(g.outdegreeMatrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.outdegreeMatrix[0].dtype, 'int32')
                self.assertEqual(g.degreeMatrix, None)
            else:
                self.assertEqual(g.degreeMatrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.degreeMatrix[0].dtype, 'int32')
                self.assertEqual(g.indegreeMatrix, None)
                self.assertEqual(g.outdegreeMatrix, None)

    def test_PMatrix(self):
        """2. Test de la matriu P
        """
        for g in self.KDA:
            P = g.compute_PMatrix(g.degreeMatrix)
            self.assertEqual(P.shape, (g.T, g.m))
    
    def test_Anonymization(self):
        """3. Test Anonimització de graus
        """
        for g in self.KDA:
            PMatrix = g.compute_PMatrix(g.degreeMatrix)
            anonymizedDegrees= g.anonymizeDegrees(g.degreeMatrix, PMatrix)
            self.assertEqual(anonymizedDegrees.shape, (g.T, g.degreeMatrix.shape[1]))
            for anonymousdegrees in anonymizedDegrees:
                unique, counts = np.unique(anonymousdegrees, return_counts=True)
                self.assertTrue(np.all(counts >= g.k))

    def test_Realizable(self):
        """4. Test matriu de graus realizable
        """
        for g in self.KDA:
            T, n = g.degreeMatrix.shape
            if g.k <= T:
                PMatrix = g.compute_PMatrix(g.degreeMatrix)
                anonymizedDegrees= g.anonymizeDegrees(g.degreeMatrix, PMatrix)
                finalMatrix = g.realizeDegrees(anonymizedDegrees)

                unique, counts = np.unique(finalMatrix, return_counts=True)
                self.assertTrue(np.all(counts >= g.k))
                for anonymousdegrees in finalMatrix:
                    unique, counts = np.unique(anonymousdegrees, return_counts=True)
                    self.assertTrue(np.all(counts >= g.k))

if __name__ == '__main__':
    unittest.main()
