import numpy as np
import networkx as nx
import os
import data_paths as dp
import pickle
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

files = ['HOUR_CollegeMsg', 
         'DAY_CollegeMsg',
         'WEEK_CollegeMsg', 
         'HOUR_ia-enron-employees',
         'DAY_ia-enron-employees',
         'WEEK_ia-enron-employees',
         'insecta-ant-colony5',
         'aves-sparrow-social', 
         'mammalia-voles-rob-trapping']

FILE = files[-1] # Posa el nom del fitxer que vols calcular/visualitzar les mètriques en cada mètode, en string

class Metrics:
    __slots__ = ()
    """Classe que computa i visualitza les mètriques entre dos grafs.
    """
    def __init__(self, mode=None):
        """Inicialitza la classe Metrics

        Args:
            mode (int, optional): Mode a executar. Mode 1: Calcula les mètriques del fitxer FILE
                                                   Mode 2: Llegeix i visualitza les mètriques càlculades
                                                   Defaults to None (No s'executa cap mode).
        """ 
        if mode == 1:
            self.computeMetrics()
        elif mode == 2:
            self.visualizeMetrics()

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
            List : Llista de densitats dels grafs originals i protegits
        """
        densityOriginal = nx.density(g1)
        densityProtected = nx.density(g2)
        return [densityOriginal, densityProtected]
    
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
        """Llegir els fitxers generats al aplicar la protecció de grafs

        Returns:
            Dict, Dict: Diccionari dels mètodes i fitxers que s'han utilitzat
        """
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
        """Calcula les mètriques de tots els fitxers que es tenen, i es guarden en format JSON.

        Raises:
            ValueError: En cas de no tenir un fitxer per calcular les mètriques
        """
        originalFiles, protectedDict = self.readMetrics()
        
        with open(originalFiles[FILE][0]+"/"+originalFiles[FILE][1], 'rb') as f:
            originalGraphs = pickle.load(f)

        for method in protectedDict.keys():
            # print(protectedDict[method].keys())
            # for file in tqdm(protectedDict[method].keys(), desc="Computing metrics"):

            if not FILE:
                raise ValueError("No file selected. Please select a file to compute metrics.")
            
            
            results = {"Jaccard": [], "DeltaConnectivity": [], "DeltaIn": [], "DeltaOut": [],
                        "Jaccard Betweenness": [], "Jaccard Closeness": [], "Jaccard Degree": [], "Densities": []}
            
            for i in tqdm(protectedDict[method][FILE], desc="Computing metrics in file: " + str(FILE) + "-" + str(method)):
                listJaccard = []
                listDeltaConnectivity = []
                listDeltaIn = []
                listDeltaOut = []
                listBetweeness = []
                listCloseness = []
                listDegree = []
                listDensities = []

                with open(i[0]+"/"+ i[1], 'rb') as f:
                    protectedGraphs = pickle.load(f)
                                
                for originalG, protectedG in zip(originalGraphs, protectedGraphs):
                    
                    if originalG.is_directed():
                        connectivityOut = self.deltaConnectivity(originalG, protectedG, 'out')
                        connectivityIn = self.deltaConnectivity(originalG, protectedG, 'in')
                        listDeltaIn.append(connectivityIn)
                        listDeltaOut.append(connectivityOut)
                    else:
                        connectivity = self.deltaConnectivity(originalG, protectedG, None)
                        listDeltaConnectivity.append(connectivity)

                    listJaccard.append(self.jaccardIndex(originalG, protectedG))
                    listBetweeness.append(self.computeCentrality(originalG, protectedG, nx.betweenness_centrality, 0.1))
                    listCloseness.append(self.computeCentrality(originalG, protectedG, nx.closeness_centrality, 0.1))
                    listDegree.append(self.computeCentrality(originalG, protectedG, nx.degree_centrality, 0.1))
                    listDensities.append(self.getDensities(originalG, protectedG))

                results["Jaccard"].append((listJaccard, i[2]))
                results["DeltaConnectivity"].append((listDeltaConnectivity, i[2]))
                results["Betweeness"].append((listBetweeness, i[2]))
                results["Closeness"].append((listCloseness, i[2]))
                results["Degree"].append((listDegree, i[2]))
                results["Densities"].append((listDensities, i[2]))

            current_dir = os.path.dirname(os.path.abspath(__file__))
            with open(current_dir + "/metrics/" + method + "/" + FILE + ".json", 'w') as f:
                json.dump(results, f)

    def viewMeanSimilarities(self):
        folders = dp.METRICS
        fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 5 * 1), squeeze=False)  # Adjust figsize as needed
        fig.tight_layout(pad=5.0)

        for idx,path in enumerate(folders):
            row, col = divmod(idx, 3)
            ax = axes[row, col]

            file = path + "/" + FILE + ".json"
            with open(file, "r") as f:
                data = json.load(f)
            
            for key in data:
                listAverages = list()
                listParameters = list()
                for metric in data[key]:
                    if key != "Densities":
                        meanMetric = np.array(metric[0]).mean()
                        parameter = metric[1]
                        listAverages.append(meanMetric)
                        listParameters.append(parameter)
                    
                if not listAverages:
                    continue

                listParameters, listAverages = zip(*sorted(zip(listParameters, listAverages)))
                # print(f"Jaccard: {listAverages}")
                # print(f"Parameters: {listParameters}")

                ax.plot(listParameters, listAverages, marker='o', label=key)
            
            if path.split("/")[-1] == "ELDP":
                xLabel =  r'$\epsilon$'
            else:
                xLabel = "k"
            ax.set_xlabel(xLabel)
            ax.set_ylabel('Percentatge de similaritat (%)')
            ax.legend()
            ax.grid(True)
            ax.set_title('Mitjana de mètriques en ' + str(FILE) + '\n Mètode de protecció: ' + str(path.split("/")[-1]))
            
            ax.set_ylim([-5, 100])
            ax.set_yticks(np.arange(0, 101, 20))

        plt.show()

    def viewIndividualSimilarities(self):
        folders = dp.METRICS
        all_metrics = ["Jaccard", "DeltaConnectivity", "Jaccard Betweenness", "Jaccard Closeness", "Jaccard Degree"]
        num_methods = len(folders)

        cols = 3
        rows = len(all_metrics) * int(np.ceil(num_methods / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5), squeeze=False)
        fig.suptitle(f"Comparativa de mètriques en: {FILE}", fontsize=12)
        fig.subplots_adjust(hspace=0.4, wspace=0.2, top=0.9, right=0.9)  # Deja espacio a la derecha para el colorbar

        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
        show_cbar = True  # Solo mostramos una vez

        for metric_idx, metric_name in enumerate(all_metrics):
            for method_idx, path in enumerate(folders):
                row = metric_idx * int(np.ceil(num_methods / cols)) + method_idx // cols
                col = method_idx % cols
                ax = axes[row][col]

                file_path = path + "/" + FILE + ".json"
                with open(file_path, "r") as f:
                    data = json.load(f)

                values_matrix = []
                param_labels = []

                for metric in data[metric_name]:
                    values = metric[0]
                    label = metric[1]
                    if values:
                        values_matrix.append(values)
                        param_labels.append(label)

                if values_matrix:
                    sorted_labels, sorted_values = zip(*sorted(zip(param_labels, values_matrix), key=lambda x: x[0]))
                    df = pd.DataFrame(sorted_values, index=sorted_labels).T

                    sns.heatmap(df, ax=ax, cbar=show_cbar, cbar_ax=cbar_ax if show_cbar else None,
                                vmin=0, vmax=100, linewidths=0.5, linecolor='gray')

                    show_cbar = False  # Ya mostramos el colorbar, no lo volvemos a mostrar

                    ax.set_title(f"{metric_name} - {path.split('/')[-1]}", fontsize=9)

                    if col == 0:
                        ax.set_ylabel("Timestamp", fontsize=8)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticks([])

                    if row == rows - 1:
                        xLabel = r'$\epsilon$' if path.split("/")[-1] == "ELDP" else "k"
                        ax.set_xlabel(xLabel, fontsize=8)
                    else:
                        ax.set_xlabel("")
                        ax.set_xticks([])

                    ax.tick_params(axis='both', labelsize=7)
                else:
                    ax.set_visible(False)

        # Etiqueta en la colorbar
        cbar_ax.set_ylabel("Similaritat (%)", fontsize=10)
        cbar_ax.tick_params(labelsize=8)
        plt.show()


    def viewDensities(self):
        """Visualitzar les densitats dels grafs originals i protegits"""
        folders = dp.METRICS

        for path in folders:
            file = path + "/" + FILE + ".json"
            with open(file, "r") as f:
                data = json.load(f)

            # Ordenar los resultados por el valor del parámetro
            sorted_results = sorted(data["Densities"], key=lambda x: x[1])

            num_plots = len(sorted_results)
            cols = 3
            rows = (num_plots + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
            fig.suptitle('Densitats en ' + str(FILE) +
                        '\nMètode de protecció: ' + str(path.split("/")[-1]), fontsize=12)
            fig.subplots_adjust(hspace=0.6, wspace=0.4, top=0.88, bottom=0.08)

            for idx, result in enumerate(sorted_results):
                row, col = divmod(idx, cols)
                ax = axes[row][col]

                listOriginal = [density[0] for density in result[0]]
                listProtected = [density[1] for density in result[0]]
                listGraphs = list(range(len(result[0])))

                parameter = result[1]
                ax.plot(listGraphs, listOriginal, marker='o', label="Graf original")
                ax.plot(listGraphs, listProtected, marker='o', label="Graf protegit")

                if row == rows - 1:
                    ax.set_xlabel("Timestamp")
                if col == 0:
                    ax.set_ylabel("Densitat")

                if path.split("/")[-1] == "ELDP":
                    typeProtection = r'$\epsilon$'
                else:
                    typeProtection = "k"
                ax.set_title(f"{typeProtection} = {parameter}")
                ax.grid(True)

            # Eliminar subplots vacíos
            for i in range(num_plots, rows * cols):
                row, col = divmod(i, cols)
                fig.delaxes(axes[row][col])

            # Añadir una única leyenda común
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=2)

            plt.show()


    def visualizeMetrics(self):
        """Visualitza totes les mètriques generades d'un fitxer.
        """
        self.viewMeanSimilarities()
        self.viewIndividualSimilarities()
        self.viewDensities()
        #! (En cas de dividir UnixTimestamps)  Fer un gràfic per cada mètrica, comparant els tres fitxers.


if __name__ == "__main__":
    metric = Metrics(mode = 2)