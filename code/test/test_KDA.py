import unittest
import sys
import os
import numpy as np
import networkx as nx

# Afegir el directori pare 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import reader as rd
import data_paths as dp
from graph import KDA

class TestKDA(unittest.TestCase):
    __slots__ = ('dictionary_options', 'readers', 'KDA', 'save', 'numExperiments')
    def setUp(self):
        """Crea una instància de KDA
        """
        # Per Testing !
        self.save = False # Canviar si es volen guardar els grafs resultants
        grouping = None
        self.dictionary_options = {'1': (dp.DATASET1, True, False, 'FILE'), 
                                    '2': (dp.DATASET3, True, True, 'FILE')} 
        setK = [3, 5]
        self.numExperiments = 1

        
        # self.save = True
        # grouping = None
        # self.dictionary_options = {'1': (dp.DATASET3, True, False, 'FILE')}
        # setK = np.arange(2,8,1)
        # self.numExperiments = 5

        self.readers = [] # Llegim els fitxers i els col·loquem en una llista
        for key, value in self.dictionary_options.items():
            self.readers.append(rd.Reader(value))
        
        self.KDA = []
        for k in setK: # Per totes les k que provem, crear una instància de KDA amb tots els fitxers
            for i,reader in enumerate(self.readers):
                self.KDA.append(KDA(reader.filename, self.dictionary_options[str(i+1)], reader.df, grouping, k))

        # nx.draw(self.ELDP[0].graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=16)
        # plt.show()

    def test_degrees(self):
        """Testing degree matrices...
        """
        for g in self.KDA:
            # En cas de ser dirigit
            if g.directed:
                self.assertEqual(g.indegreeMatrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.indegreeMatrix.dtype, 'int32')
                self.assertEqual(g.indegreeMatrix.shape, (g.T, g.graph.number_of_nodes()))
                self.assertEqual(g.outdegreeMatrix.shape, (g.T, g.graph.number_of_nodes()))
                self.assertEqual(g.outdegreeMatrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.outdegreeMatrix.dtype, 'int32')
                self.assertEqual(g.degreeMatrix, None)
            # En cas de no ser dirigit
            else:
                self.assertEqual(g.degreeMatrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.degreeMatrix.dtype, 'int32')
                self.assertEqual(g.degreeMatrix.shape, (g.T, g.graph.number_of_nodes()))
                self.assertEqual(g.indegreeMatrix, None)
                self.assertEqual(g.outdegreeMatrix, None)

    def test_PMatrix(self):
        """Testing PMatrix...
        """
        for g in self.KDA:
            # En cas de no ser dirigit
            if not g.directed:
                P = g.compute_PMatrix(g.degreeMatrix)
                self.assertEqual(P.shape, (g.T, g.m))
                self.assertEqual(P.dtype, 'int32')
            # En cas de ser dirigit
            else:
                PMatrixIn = g.compute_PMatrix(g.indegreeMatrix)
                self.assertEqual(PMatrixIn.shape, (g.T, g.m))
                self.assertEqual(PMatrixIn.dtype, 'int32')

    def test_Anonymization(self):
        """Testing Anonimització de graus...
        """
        for g in self.KDA:
            # En cas de no ser dirigit
            if not g.directed:
                PMatrix = g.compute_PMatrix(g.degreeMatrix)
                anonymizedDegrees= g.anonymizeDegrees(g.degreeMatrix, PMatrix)
                # Comprovem que sigui de la mateixa dimensió la matriu 
                self.assertEqual(anonymizedDegrees.shape, (g.T, g.graph.number_of_nodes()))
                for anonymousdegrees in anonymizedDegrees:
                    unique, counts = np.unique(anonymousdegrees, return_counts=True)
                    # Veiem que per cada seqüència (graf individual), sigui k-anònim
                    self.assertTrue(np.all(counts >= g.k))
            # En cas de ser dirigit
            else:
                PMatrixIn = g.compute_PMatrix(g.indegreeMatrix)
                anonymizedDegreesIn= g.anonymizeDegrees(g.indegreeMatrix, PMatrixIn)
                # Comprovem que sigui de la mateixa dimensió la matriu 
                self.assertEqual(anonymizedDegreesIn.shape, (g.T, g.graph.number_of_nodes()))

                for indegrees in zip(anonymizedDegreesIn):
                    uniqueIn, countsIn = np.unique(indegrees, return_counts=True)
                    # Veiem que per cada seqüència (graf individual), sigui k-anònim
                    self.assertTrue(np.all(countsIn >= g.k))

    def test_Realizable(self):
        """Testing matriu de graus realizable...
        """
        for g in self.KDA:
            if g.k <= g.T:
                # En cas de no ser dirigit
                if not g.directed:
                    PMatrix = g.compute_PMatrix(g.degreeMatrix)
                    anonymizedDegrees= g.anonymizeDegrees(g.degreeMatrix, PMatrix)
                    finalMatrix = g.realizeDegrees(anonymizedDegrees)
                    # Comprovem que ara la matriu compleix K-anonimitat (que les seqüències graficables apareguin k vegades com a mínim)
                    unique, counts = np.unique(finalMatrix, return_counts=True)
                    self.assertTrue(np.all(counts >= g.k))
                    for anonymousdegrees in finalMatrix:
                        unique, counts = np.unique(anonymousdegrees, return_counts=True)
                        self.assertTrue(np.all(counts >= g.k))
                
                # En cas de ser dirigit
                else:
                    PMatrixIn = g.compute_PMatrix(g.indegreeMatrix)
                    anonymizedDegreesIn= g.anonymizeDegrees(g.indegreeMatrix, PMatrixIn)
                    finalMatrixIn = g.realizeDegrees(anonymizedDegreesIn)

                    uniqueIn, countsIn = np.unique(finalMatrixIn, return_counts=True)
                    # Comprovem que ara la matriu compleix K-anonimitat (que les seqüències graficables apareguin k vegades com a mínim)
                    self.assertTrue(np.all(countsIn >= g.k))
                    for indegrees in finalMatrixIn:
                        uniqueIn, countsIn = np.unique(indegrees, return_counts=True)
                        self.assertTrue(np.all(countsIn >= g.k))
        
    def test_Construction(self):
        """Testing Havel-Hakimi...
        """
        for g in self.KDA:
            if g.k <= g.T:
                # En cas de no ser dirigit
                if not g.directed:
                    PMatrix = g.compute_PMatrix(g.degreeMatrix)
                    anonymizedDegrees= g.anonymizeDegrees(g.degreeMatrix, PMatrix)
                    finalMatrix = g.realizeDegrees(anonymizedDegrees)
                    for row in finalMatrix:
                        graph = g.createProtectedGraphUndirected(row)
                        finalDict = dict(graph.degree())
                        finalList = np.array([finalDict.get(node, 0) for node in graph.nodes()])
                        self.assertEqual(row.shape, finalList.shape)
                        self.assertTrue(np.array_equal(finalList, row))

                # En cas de ser dirigit
                else:
                    PMatrixIn = g.compute_PMatrix(g.indegreeMatrix)
                    anonymizedDegreesIn= g.anonymizeDegrees(g.indegreeMatrix, PMatrixIn)
                    finalMatrixIn = g.realizeDegrees(anonymizedDegreesIn)

                    for indegrees in finalMatrixIn:
                        # Fem que la matriu de indegrees sigui la mateixa que la de outdegrees, però permutada
                        # ÉS NECESSARI QUE LA SUMA DE OUTDEGREES SIGUI IGUAL A LA DE INDEGREES - Per això es fa la permutació
                        outdegrees = indegrees.copy()
                        outdegrees = np.random.permutation(indegrees)
                        graph = g.createProtectedGraphDirected(indegrees, outdegrees)
                        finalInDict = dict(graph.in_degree())
                        finalInList = np.array([finalInDict.get(node, 0) for node in graph.nodes()])
                        finalOutDict = dict(graph.out_degree())
                        finalOutList = np.array([finalOutDict.get(node, 0) for node in graph.nodes()])
                        self.assertEqual(np.sum(indegrees), np.sum(outdegrees))
                        self.assertEqual(indegrees.shape, finalInList.shape)
                        self.assertTrue(np.array_equal(finalInList, indegrees))
                        self.assertEqual(outdegrees.shape, finalOutList.shape)
                        self.assertTrue(np.array_equal(finalOutList, outdegrees))
                        
    def test_protection(self):
        """Testing protecció KDA...
        """
        for e in range(self.numExperiments):
            for g in self.KDA:
                originalList, protectedList = g.apply_protection(randomize = True)

                self.assertIsInstance(originalList, list)
                self.assertIsInstance(protectedList, list)

                if len(originalList) > 0 and len(protectedList) > 0:
                    self.assertIsInstance(originalList[0], nx.Graph)
                    self.assertIsInstance(protectedList[0], nx.Graph)
                
                self.assertEqual(len(originalList), len(protectedList))

                if self.save and len(originalList) > 0 and len(protectedList) > 0:
                    g.save_graphs(originalList, protectedList, "KDA", e+1, g.k)


if __name__ == '__main__':
    unittest.main()
