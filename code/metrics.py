import numpy as np
import networkx as nx
class Metrics:
    __slots__ = ('originalGraphs', 'protectedGraphs')
    """Classe que computa i visualitza les mètriques entre dos grafs.
    """
    def __init__(self, originalGraphs, protectedGraphs):
        """Inicialitza la classe Metrics amb els grafs originals i protegits.

        Args:
            originalGraphs (List): Grafs originals 
            protectedGraphs (List): Grafs protegits per un algorisme de protecció
        """
        self.originalGraphs = originalGraphs
        self.protectedGraphs = protectedGraphs
    
    def edgeIntersection(self, e1, e2):
        """Calcular la intersecció d'arestes de dos grafs

        Args:
            e1, e2 (set): Conjunt d'arestes d'un graf

        Returns:
            int: Nombre d'elements en la intersecció
        """
        intersection = e1 & e2
        # print(f"Intersection: {intersection}")
        return len(intersection)
    
    def edgeUnion(self, e1, e2):
        """Calcular l'unió d'arestes de dos grafs

        Args:
            e1, e2 (set): Conjunt d'arestes d'un graf

        Returns:
            int: Nomnre d'elements en l'unió
        """
        union = e1 | e2
        # print(f"Union: {union}")
        return len(union)

    def jaccardCoefficient(self, g1, g2):
        # En cas de no ser dirigit els grafs, s'ha de descartar les arestes duplicades
        if g1.is_directed():
            edgesG1 = set(g1.edges())
            edgesG2 = set(g2.edges())
        else:
            edgesG1 = set(tuple(sorted(edge)) for edge in g1.edges())
            edgesG2 = set(tuple(sorted(edge)) for edge in g2.edges())
        
        # print(f"Edges G1: {edgesG1}, Edges G2: {edgesG2}")

        intersection = self.edgeIntersection(edgesG1, edgesG2)
        union = self.edgeUnion(edgesG1, edgesG2)
 
        print()

        # Quan un dels dos grafs són buits o els dos són buits
        if union == 0:
            return 1.0 if intersection == 0 else 0.0  

        # Calculem el coeficient de Jaccard i el retornem en forma de percentatge
        return (intersection / union)*100

    def degreeMatrices(self, graph):
        """Obtenir les matrius diagonals de graus del graf

        Args:
            graph (nx.Graph): Graf a calcular la seva matriu de graus diagonals 

        Returns:
            np.array, int: Matriu diagonal de graus, amb el grau màxim
        """
        # Obtenim les seqüències de grau per cada graf, i calculem el grau màxim
        degreeDict1 = dict(graph.degree())
        degreeSequence1 = [degreeDict1.get(node, 0) for node in graph.nodes()]
        maxDegree1 = max(degreeSequence1)

        # Ho tornem en una matriu diagonal i la retornem 
        degreeMatrix1 = np.diag(degreeSequence1)

        return degreeMatrix1, maxDegree1

    def influenceNeighbors(self, maxDegree):
        """Calcular la influència entre veïns

        Args:
            maxDegree (int): Grau màxim del graf

        Returns:
            float: Influència entre veïns
        """
        return 1 / (1+maxDegree)

    def scoreMatrix(self, graph):
        identityMatrix = np.identity(graph.number_of_nodes())
        # print(f"Identity Matrix: {identityMatrix}")
        degreeMatrix, maxDegree = self.degreeMatrices(graph)
        adjacencyMatrix = nx.adjacency_matrix(graph).toarray()
        # print(f"Adjacency Matrix: {adjacencyMatrix}")
        influence = self.influenceNeighbors(maxDegree)
        squaredInfluence = pow(self.influenceNeighbors(maxDegree), 2) 
        finalMatrix = identityMatrix + (squaredInfluence * degreeMatrix) - (influence * adjacencyMatrix)
        
    def rootEuclideanDistance(self):
        pass

    def deltaConnectivity(self, g1, g2):
        # Aplicar l'algorisme per grafs no dirigits
        if not g1.is_directed() and not g2.is_directed():
            pass
        
    
    def spectralSimilarity(self, g1, g2):
        pass

    def centralityCorrelation(self, g1, g2, function):
        pass
    
    def computeMetrics(self):
        pass

    def visualizeMetrics(self):
        pass