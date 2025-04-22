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

    def jaccardIndex(self, g1, g2):
        """Calcular el coeficient de Jaccard entre dos grafs (edge overlap)

        Args:
            g1, g2 (nx.Graph): Grafs a calcular el coeficient de Jaccard

        Returns:
            float: Coeficient de Jaccard entre els dos grafs, valor en percentatge
        """
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

    def degreeMatrices(self, graph, type):
        """Obtenir les matrius diagonals de graus del graf

        Args:
            graph (nx.Graph): Graf a calcular la seva matriu de graus diagonals 

        Returns:
            np.array, int: Matriu diagonal de graus, amb el grau màxim
        """
        # Obtenim les seqüències de grau per cada graf, i calculem el grau màxim
        if graph.is_directed() and type == 'out':
            degreeDict1 = dict(graph.out_degree())
        elif graph.is_directed() and type == 'in':
            degreeDict1 = dict(graph.in_degree())
        else:
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

    def scoreMatrix(self, graph, type):
        """Calcular la matriu d'influència de nodes d'un graf

        Args:
            graph (nx.Graph): Graf a calcular la matriu S

        Returns:
            np.array: Matriu S, que descriu la influència entre nodes
        """
        identityMatrix = np.identity(graph.number_of_nodes())
        # print(f"Identity Matrix: {identityMatrix}")
        degreeMatrix, maxDegree = self.degreeMatrices(graph, type)
        adjacencyMatrix = nx.adjacency_matrix(graph).toarray()
        # print(f"Adjacency Matrix: {adjacencyMatrix}")
        influence = self.influenceNeighbors(maxDegree)
        squaredInfluence = pow(self.influenceNeighbors(maxDegree), 2) 
        S = identityMatrix + (squaredInfluence * degreeMatrix) - (influence * adjacencyMatrix)
        finalMatrix = np.linalg.inv(S)
        # print(f"Final Matrix: {finalMatrix}")
        return finalMatrix
    
    def rootEuclideanDistance(self, S1, S2):
        """Calcular RootED de les matrius d'influència.

        Args:
            S1, S2 (np.array): Matrius d'influència de dos grafs
        
        Returns:
            float: valor numèric que defineix la distància entre les dues matrius
        """
        diff = S1 - S2
        squared_diff = np.square(diff)
        total_sum = np.sum(squared_diff)
        distance = np.sqrt(total_sum)
        return distance

    def deltaConnectivity(self, g1, g2, type):
        """Obtenir la similaritat per l'afinitat entre nodes de dos grafs

        Args:
            g1, g2 (nx.Graph): Grafs a calcular la seva similaritat

        Returns:
            float: Similaritat entre els dos grafs (afinitat), valor en percentatge
        """
        S1 = self.scoreMatrix(g1, type)
        S2 = self.scoreMatrix(g2, type)
        # print(f"S1: {S1}, S2: {S2}")
        distance = self.rootEuclideanDistance(S1, S2)
        # print(f"Distance: {distance}")
        return (1/(1+distance))*100
        
    def getDensities(self, g1, g2):
        """Obtenir les densitats dels grafs originals i protegits

        Args:
            g1, g2 (nx.Graph): Grafs a calcular la seva densitat

        Returns:
            List, List: Llista de densitats dels grafs originals i protegits
        """
        densityOriginal = [nx.density(g) for g in g1]
        densityProtected = [nx.density(g) for g in g2]
        return densityOriginal, densityProtected
    
    def topKNodes(centralityDict, k):
        """Obtenir els k nodes més centrals d'un graf

        Args:
            centrality_dict (Dict): Diccionari de centralitat dels nodes
            k (int): nombre de nodes més centrals a obtenir

        Returns:
            set: Conjunt dels k nodes més centrals
        """
        return set(sorted(centralityDict, key=centralityDict.get, reverse=True)[:k])

    def computeCentrality(self, g1, g2, function):
        centrality1 = function(g1)
        centrality2 = function(g2)

        # Obtenim el 5% nodes més centrals de cada graf
        k = max(1, int(0.05 * g1.number_of_nodes()))
        topG1 = self.topKNodes(centrality1, k)
        topG2 = self.topKNodes(centrality2, k)
        
        intersection = topG1 & topG2
        union = topG1 | topG2
        return len(intersection) / len(union)
    
    def getCentrality(self, g1, g2):
        jaccardBC = self.computeCentrality(g1, g2, nx.betweenness_centrality)
        jaccardCC = self.computeCentrality(g1, g2, nx.closeness_centrality)
        jaccardDC = self.computeCentrality(g1, g2, nx.degree_centrality)
        return jaccardBC, jaccardCC, jaccardDC
        
    def computeMetrics(self):
        pass

    def visualizeMetrics(self):
        pass