import numpy as np
import networkx as nx
import os
import data_paths as dp
import pickle
from tqdm import tqdm
import json

FILE = None # Posa el nom del fitxer que vols calcular les mètriques en cada mètode, en string

class Metrics:
    __slots__ = ()
    """Classe que computa i visualitza les mètriques entre dos grafs.
    """
    def __init__(self):
        """Inicialitza la classe Metrics amb els grafs originals i protegits.
        """
        #! Fer un seleccionador, per tal de voler calcular o visualitzar les mètriques
        #self.computeMetrics()
        # self.visualizeMetrics()

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
 
        # print()

        # Quan un dels dos grafs són buits o els dos són buits
        if union == 0:
            return 1.0 if intersection == 0 else 0.0  

        # Calculem el coeficient de Jaccard i el retornem en forma de percentatge
        return (intersection / union)*100

    def degreeMatrices(self, graph, type):
        """Obtenir les matrius diagonals de graus del graf

        Args:
            graph (nx.Graph): Graf a calcular la seva matriu de graus diagonals 
            type (str): tipus de connexió a calcular (in, out, None)
            
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
            type (str): tipus de connexió a calcular (in, out, None)

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
            type (str): tipus de connexió a calcular (in, out, None)
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
    
    def topKNodes(self, centralityDict, k):
        """Obtenir els k nodes més centrals d'un graf

        Args:
            centrality_dict (Dict): Diccionari de centralitat dels nodes
            k (int): nombre de nodes més centrals a obtenir

        Returns:
            set: Conjunt dels k nodes més centrals
        """
        return set(sorted(centralityDict, key=centralityDict.get, reverse=True)[:k])

    def computeCentrality(self, g1, g2, function, topNodes):
        """Calcular índex de Jaccard per les centralitats entre dos grafs

        Args:
            g1, g2 (nx.Graph): Grafs a calcular la seva centralitat
            function (nx): Funció de centralitat a aplicar (betweenness, closeness, degree)	
            topNodes (float): Percentatge de nodes més centrals a obtenir   
        Returns:
            float: Índex de Jaccard entre els dos grafs, valor en percentatge
        """
        centrality1 = function(g1)
        centrality2 = function(g2)

        # Obtenim el % de nodes més centrals de cada graf
        k = max(1, int(topNodes * g1.number_of_nodes()))
        topG1 = self.topKNodes(centrality1, k)
        topG2 = self.topKNodes(centrality2, k)
        
        intersection = topG1 & topG2
        union = topG1 | topG2
        return (len(intersection) / len(union))*100
    
    def getCentrality(self, g1, g2, topNodes):
        """Obtenir les mètriques de centralitat entre dos grafs

        Args:
            g1, g2 (nx.Graph): Grafs a calcular la seva centralitat
            topNodes (float): valor entre 0 i 1 que defineix el % de nodes més centrals a obtenir

        Returns:
            float, float, float: Índex de Jaccard entre els dos grafs de totes les funcions de centralitat, valor en percentatge
        """
        jaccardBC = self.computeCentrality(g1, g2, nx.betweenness_centrality, topNodes)
        jaccardCC = self.computeCentrality(g1, g2, nx.closeness_centrality, topNodes)
        jaccardDC = self.computeCentrality(g1, g2, nx.degree_centrality, topNodes)
        return jaccardBC, jaccardCC, jaccardDC
    
    def readMetrics(self):
        folders = dp.OUTPUTS
        originalFiles = {}
        protectedDict = {}

        for root_folder in folders:
            name = root_folder.split("/")[-1]
            protectedDict[name] = {}
            for dirpath, dirnames, filenames in os.walk(root_folder):
                if name != "original_graphs":
                    for f in filenames:
                        nameFile = f.split(".")[0]  
                        if nameFile not in protectedDict[name]:
                            protectedDict[name][nameFile] = []
                        protectedDict[name][nameFile].append((dirpath, f, float(f.split("_")[-1][:-4])))
                else:
                    protectedDict.pop(name)
                    for f in filenames:
                        nameFile = f.split(".")[0]  
                        originalFiles[nameFile] = (dirpath, f)

        # print(f"Original files: {originalFiles}")
        # print(f"Protected files: {protectedDict}")        
        return originalFiles, protectedDict

    def computeMetrics(self):
        originalFiles, protectedDict = self.readMetrics()
        for method in protectedDict.keys():
            print(protectedDict[method].keys())
            # for file in tqdm(protectedDict[method].keys(), desc="Computing metrics"):

            if not FILE:
                raise ValueError("No file selected. Please select a file to compute metrics.")
            
            with open(originalFiles[FILE][0]+"/"+originalFiles[FILE][1], 'rb') as f:
                originalGraphs = pickle.load(f)
            
            results = {"Jaccard": [], "DeltaConnectivity": [], "Betweeness": [], "Closeness": [], "Degree": []}
            for i in tqdm(protectedDict[method][FILE], desc="Computing metrics in file: " + str(FILE)):
                listJaccard = []
                listDeltaConnectivity = []
                listBetweeness = []
                listCloseness = []
                listDegree = []

                with open(i[0]+"/"+ i[1], 'rb') as f:
                    protectedGraphs = pickle.load(f)
                                
                for originalG, protectedG in zip(originalGraphs, protectedGraphs):
                    
                    if originalG.is_directed():
                        connectivity = self.deltaConnectivity(originalG, protectedG, 'out')
                    else:
                        connectivity = self.deltaConnectivity(originalG, protectedG, None)
                        
                    listJaccard.append(self.jaccardIndex(originalG, protectedG))
                    listDeltaConnectivity.append(connectivity)
                    listBetweeness.append(self.computeCentrality(originalG, protectedG, nx.betweenness_centrality, 0.1))
                    listCloseness.append(self.computeCentrality(originalG, protectedG, nx.closeness_centrality, 0.1))
                    listDegree.append(self.computeCentrality(originalG, protectedG, nx.degree_centrality, 0.1))

                results["Jaccard"].append((listJaccard, i[2]))
                results["DeltaConnectivity"].append((listDeltaConnectivity, i[2]))
                results["Betweeness"].append((listBetweeness, i[2]))
                results["Closeness"].append((listCloseness, i[2]))
                results["Degree"].append((listDegree, i[2]))

            current_dir = os.path.dirname(os.path.abspath(__file__))
            with open(current_dir + "/metrics/" + method + "/" + FILE + ".json", 'w') as f:
                json.dump(results, f)

    def visualizeMetrics(self):
        folders = dp.METRICS
        for root_folder in folders:
            for dirpath, dirnames, filenames in os.walk(root_folder):
                for f in filenames:
                    with open(dirpath + "/" + f, 'r') as file:
                        data = json.load(file)
                    
                    print(f"Data: {data}")
                    break
            break
        

if __name__ == "__main__":
    metric = Metrics()