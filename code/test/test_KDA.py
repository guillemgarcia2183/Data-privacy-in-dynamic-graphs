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
    __slots__ = ('dictionary_options', 'readers', 'KDA', 'save')
    def setUp(self):
        """Crea una instància de KDA
        """
        self.save = False # Canviar si es volen guardar els grafs resultants

        self.dictionary_options = {'1': (dp.DATASET1, True, False, 'FILE'),
                                   '2': (dp.DATASET2, True, False, 'FILE'), 
                               '3': (dp.DATASET3, True, True, 'FILE'),}
        
        self.readers = []
        for key, value in self.dictionary_options.items():
            self.readers.append(rd.Reader(value))
        
        # setK = np.arange(2, 15, 1)
        setK = np.arange(2, 7, 1)
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
            # En cas de ser dirigit
            if g.directed:
                self.assertEqual(g.indegreeMatrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.indegreeMatrix[0].dtype, 'int32')
                self.assertEqual(g.outdegreeMatrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.outdegreeMatrix[0].dtype, 'int32')
                self.assertEqual(g.degreeMatrix, None)
            # En cas de no ser dirigit
            else:
                self.assertEqual(g.degreeMatrix[0].shape, (g.graph.number_of_nodes(),))
                self.assertEqual(g.degreeMatrix[0].dtype, 'int32')
                self.assertEqual(g.indegreeMatrix, None)
                self.assertEqual(g.outdegreeMatrix, None)

    def test_PMatrix(self):
        """2. Test de la matriu P
        """
        for g in self.KDA:
            # En cas de no ser dirigit
            if not g.directed:
                P = g.compute_PMatrix(g.degreeMatrix)
                self.assertEqual(P.shape, (g.T, g.m))
            # En cas de ser dirigit
            else:
                PMatrixIn = g.compute_PMatrix(g.indegreeMatrix)
                PMatrixOut = g.compute_PMatrix(g.outdegreeMatrix)
                self.assertEqual(PMatrixIn.shape, (g.T, g.m))
                self.assertEqual(PMatrixOut.shape, (g.T, g.m))
            
    def test_Anonymization(self):
        """3. Test Anonimització de graus
        """
        for g in self.KDA:
            # En cas de no ser dirigit
            if not g.directed:
                PMatrix = g.compute_PMatrix(g.degreeMatrix)
                anonymizedDegrees= g.anonymizeDegrees(g.degreeMatrix, PMatrix)
                self.assertEqual(anonymizedDegrees.shape, (g.T, g.degreeMatrix.shape[1]))
                for anonymousdegrees in anonymizedDegrees:
                    unique, counts = np.unique(anonymousdegrees, return_counts=True)
                    self.assertTrue(np.all(counts >= g.k))
            # En cas de ser dirigit
            else:
                PMatrixIn = g.compute_PMatrix(g.indegreeMatrix)
                PMatrixOut = g.compute_PMatrix(g.outdegreeMatrix)
                anonymizedDegreesIn= g.anonymizeDegrees(g.indegreeMatrix, PMatrixIn)
                anonymizedDegreesOut= g.anonymizeDegrees(g.outdegreeMatrix, PMatrixOut)
                self.assertEqual(anonymizedDegreesIn.shape, (g.T, g.indegreeMatrix.shape[1]))
                self.assertEqual(anonymizedDegreesOut.shape, (g.T, g.outdegreeMatrix.shape[1]))

                for indegrees, outdegrees in zip(anonymizedDegreesIn, anonymizedDegreesOut):
                    uniqueIn, countsIn = np.unique(indegrees, return_counts=True)
                    uniqueOut, countsOut = np.unique(outdegrees, return_counts=True)
                    self.assertTrue(np.all(countsIn >= g.k))
                    self.assertTrue(np.all(countsOut >= g.k))


    def test_Realizable(self):
        """4. Test matriu de graus realizable
        """
        for g in self.KDA:
            try:
                T, n = g.degreeMatrix.shape
            except:
                T, n = g.indegreeMatrix.shape

            if g.k <= T:
                # En cas de no ser dirigit
                if not g.directed:
                    PMatrix = g.compute_PMatrix(g.degreeMatrix)
                    anonymizedDegrees= g.anonymizeDegrees(g.degreeMatrix, PMatrix)
                    finalMatrix = g.realizeDegrees(anonymizedDegrees)

                    unique, counts = np.unique(finalMatrix, return_counts=True)
                    self.assertTrue(np.all(counts >= g.k))
                    for anonymousdegrees in finalMatrix:
                        unique, counts = np.unique(anonymousdegrees, return_counts=True)
                        self.assertTrue(np.all(counts >= g.k))
                
                # En cas de ser dirigit
                else:
                    PMatrixIn = g.compute_PMatrix(g.indegreeMatrix)
                    PMatrixOut = g.compute_PMatrix(g.outdegreeMatrix)
                    anonymizedDegreesIn= g.anonymizeDegrees(g.indegreeMatrix, PMatrixIn)
                    anonymizedDegreesOut= g.anonymizeDegrees(g.outdegreeMatrix, PMatrixOut)
                    finalMatrixIn = g.realizeDegrees(anonymizedDegreesIn)
                    finalMatrixOut = g.realizeDegrees(anonymizedDegreesOut)

                    uniqueIn, countsIn = np.unique(finalMatrixIn, return_counts=True)
                    uniqueOut, countsOut = np.unique(finalMatrixOut, return_counts=True)
                    self.assertTrue(np.all(countsIn >= g.k))
                    self.assertTrue(np.all(countsOut >= g.k))
                    for indegrees, outdegrees in zip(finalMatrixIn, finalMatrixOut):
                        uniqueIn, countsIn = np.unique(indegrees, return_counts=True)
                        uniqueOut, countsOut = np.unique(outdegrees, return_counts=True)
                        self.assertTrue(np.all(countsIn >= g.k))
                        self.assertTrue(np.all(countsOut >= g.k))
        
    def test_Construction(self):
        """5. Test construcció nou graf
        """
        for g in self.KDA:
            try:
                T, n = g.degreeMatrix.shape
            except:
                T, n = g.indegreeMatrix.shape
            if g.k <= T:
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
                else:
                    # ÉS POT FER AIXÍ? EN TEORIA LES MEDIANES SÓN IGUALS, NOMÉS QUE DIFEREIXEN EN EL MOMENT DE FER LES PARTICIONS
                    PMatrixIn = g.compute_PMatrix(g.indegreeMatrix)
                    anonymizedDegreesIn= g.anonymizeDegrees(g.indegreeMatrix, PMatrixIn)
                    finalMatrixIn = g.realizeDegrees(anonymizedDegreesIn)

                    for indegrees in finalMatrixIn:
                        outdegrees = indegrees.copy()
                        # ÉS NECESSARI QUE LA SUMA DE OUTDEGREES SIGUI IGUAL A LA DE INDEGREES
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
        """6. Test protecció del graf
        """
        for g in self.KDA:
            if not g.directed:
                originalList, protectedList = g.protectionUndirected()
            else:
                originalList, protectedList = g.protectionDirected()

            self.assertIsInstance(originalList, list)
            self.assertIsInstance(protectedList, list)

            if len(originalList) > 0 and len(protectedList) > 0:
                self.assertIsInstance(originalList[0], nx.Graph)
                self.assertIsInstance(protectedList[0], nx.Graph)
            
            self.assertEqual(len(originalList), len(protectedList))

            if self.save and len(originalList) > 0 and len(protectedList) > 0:
                g.save_graphs(originalList, protectedList, "KDA", g.k)



if __name__ == '__main__':
    unittest.main()
