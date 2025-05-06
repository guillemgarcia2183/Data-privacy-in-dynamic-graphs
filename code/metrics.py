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
import math

files = ['HOUR_CollegeMsg', 
         'DAY_CollegeMsg',
         'WEEK_CollegeMsg',
         'MONTH_CollegeMsg',
         'HOUR_ia-enron-employees',
         'DAY_ia-enron-employees',
         'WEEK_ia-enron-employees',
         'MONTH_ia-enron-employees',
         'insecta-ant-colony5',
         'aves-sparrow-social', 
         'mammalia-voles-rob-trapping']

FILE = files[-2] # Posa el nom del fitxer que vols calcular/visualitzar les mètriques en cada mètode, en string

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

    def degreeMatrices(self, graph):
        """Obtenir les matrius diagonals de graus del graf

        Args:
            graph (nx.Graph): Graf a calcular la seva matriu de graus diagonals 
            
        Returns:
            np.array, int: Matriu diagonal de graus, amb el grau màxim
        """
        # Obtenim les seqüències de grau per cada graf, i calculem el grau màxim 
        degreeDict= dict(graph.degree())
        degreeSequence = [degreeDict.get(node, 0) for node in graph.nodes()]
        maxDegree = max(degreeSequence)

        # Ho tornem en una matriu diagonal i la retornem 
        degreeMatrix = np.diag(degreeSequence)

        return degreeMatrix, maxDegree

    def influenceNeighbors(self, maxDegree):
        """Calcular la influència entre veïns

        Args:
            maxDegree (int): Grau màxim del graf

        Returns:
            float: Influència entre veïns
        """
        return 1 / (1+maxDegree)

    def scoreMatrix(self, graph):
        """Calcular la matriu d'influència de nodes d'un graf

        Args:
            graph (nx.Graph): Graf a calcular la matriu S

        Returns:
            np.array: Matriu S, que descriu la influència entre nodes
        """
        identityMatrix = np.identity(graph.number_of_nodes())
        # print(f"Identity Matrix: {identityMatrix}")
        degreeMatrix, maxDegree = self.degreeMatrices(graph)
        adjacencyMatrix = nx.to_numpy_array(graph)
        # print(f"Adjacency Matrix: {adjacencyMatrix}")
        influence = self.influenceNeighbors(maxDegree)
        S = identityMatrix + ((influence**2) * degreeMatrix) - (influence * adjacencyMatrix)
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
        rootS1 = np.sqrt(np.abs(S1))
        rootS2 = np.sqrt(np.abs(S2))
        diff = rootS1 - rootS2
        squared_diff = np.square(diff)
        total_sum = np.sum(squared_diff)
        distance = math.sqrt(total_sum)
        return distance

    def deltaConnectivity(self, g1, g2):
        """Obtenir la similaritat per l'afinitat entre nodes de dos grafs

        Args:
            g1, g2 (nx.Graph): Grafs a calcular la seva similaritat
            type (str): tipus de connexió a calcular (in, out, None)
        Returns:
            float: Similaritat entre els dos grafs (afinitat), valor en percentatge
        """
        S1 = self.scoreMatrix(g1)
        S2 = self.scoreMatrix(g2)
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
    
    def getDegrees(self, g1, g2):
        """Obtenir la mitjana i mediana de graus dels grafs d'entrada

        Args:
            g1, g2 (nx.Graph): Grafs a calcular els seus graus
        
        Returns:
            List: Llista de la mitjana i mediana de graus de cada graf
        """
        degreesG1 = [grau for _, grau in g1.degree()]
        degreesG2 = [grau for _, grau in g2.degree()]
        
        meanDegreeG1 = np.mean(degreesG1)
        medianDegreeG1 = np.median(degreesG1)

        meanDegreeG2 = np.mean(degreesG2)
        medianDegreeG2 = np.median(degreesG2)

        return [(meanDegreeG1, medianDegreeG1), (meanDegreeG2, medianDegreeG2)]


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
            
            
            results = {"Jaccard": [], "DeltaConnectivity": [], "Jaccard Betweenness": [], 
                       "Jaccard Closeness": [], "Jaccard Degree": [], "Densities": [], "Degrees": []}
            
            try:
                for i in tqdm(protectedDict[method][FILE], desc="Computing metrics in file: " + str(FILE) + "-" + str(method)):
                    listJaccard = []
                    listDeltaConnectivity = []
                    listBetweeness = []
                    listCloseness = []
                    listDegreeCentrality = []
                    listDensities = []
                    listDegrees = []

                    with open(i[0]+"/"+ i[1], 'rb') as f:
                        protectedGraphs = pickle.load(f)
                                    
                    for originalG, protectedG in zip(originalGraphs, protectedGraphs):
                        listJaccard.append(self.jaccardIndex(originalG, protectedG))
                        listDeltaConnectivity.append(self.deltaConnectivity(originalG, protectedG))
                        listBetweeness.append(self.computeCentrality(originalG, protectedG, nx.betweenness_centrality, 0.05))
                        listCloseness.append(self.computeCentrality(originalG, protectedG, nx.closeness_centrality, 0.05))
                        listDegreeCentrality.append(self.computeCentrality(originalG, protectedG, nx.degree_centrality, 0.05))
                        listDensities.append(self.getDensities(originalG, protectedG))
                        listDegrees.append(self.getDegrees(originalG, protectedG))

                    results["Jaccard"].append((listJaccard, i[2]))
                    results["DeltaConnectivity"].append((listDeltaConnectivity, i[2]))
                    results["Jaccard Betweenness"].append((listBetweeness, i[2]))
                    results["Jaccard Closeness"].append((listCloseness, i[2]))
                    results["Jaccard Degree"].append((listDegreeCentrality, i[2]))
                    results["Densities"].append((listDensities, i[2]))
                    results["Degrees"].append((listDegrees, i[2]))

                current_dir = os.path.dirname(os.path.abspath(__file__))
                with open(current_dir + "/metrics/" + method + "/" + FILE + ".json", 'w') as f:
                    json.dump(results, f)
            except:
                pass

    def viewMeanSimilarities(self):
        folders = dp.METRICS
        fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 5 * 1), squeeze=False)  # Adjust figsize as needed
        fig.tight_layout(pad=5.0)

        for idx,path in enumerate(folders):
            row, col = divmod(idx, 3)
            ax = axes[row, col]

            file = path + "/" + FILE + ".json"
            try:
                with open(file, "r") as f:
                    data = json.load(f)
            except:
                continue
            
            listMetrics = list()
            listKeys = list()
            for key in data:
                listAverages = list()
                listParameters = list()
                for metric in data[key]:
                    if key not in ["Densities", "Degrees"]:
                        meanMetric = np.array(metric[0]).mean()
                        parameter = metric[1]
                        listAverages.append(meanMetric)
                        listParameters.append(parameter)
                        
                        if FILE == "aves-sparrow-social" and path.split("/")[-1] != "ELDP":
                            listMetrics.append(meanMetric)
                            listKeys.append(key)

                if not listAverages:
                    continue

                listParameters, listAverages = zip(*sorted(zip(listParameters, listAverages)))
                # print(f"Jaccard: {listAverages}")
                # print(f"Parameters: {listParameters}")
                if FILE == "aves-sparrow-social" and path.split("/")[-1] != "ELDP":
                    continue
                else:
                    ax.plot(listParameters, listAverages, marker='o', label=key)
            
            if FILE == "aves-sparrow-social" and path.split("/")[-1] != "ELDP": 
                listMetrics, listKeys = zip(*sorted(zip(listMetrics, listKeys), reverse=True))
                ax.bar(listKeys, listMetrics, color="blue")
                ax.tick_params(axis='x', labelrotation=45)
                ax.set_axisbelow(True)  

            title = 'Mitjana de mètriques en ' + str(FILE) + '\n Mètode de protecció: ' + str(path.split("/")[-1])
            if path.split("/")[-1] == "ELDP":
                xLabel =  r'$\epsilon$'
            elif path.split("/")[-1] != "ELDP" and FILE == "aves-sparrow-social":
                xLabel = "Mètriques de similaritat"
                title = 'Mitjana de mètriques en ' + str(FILE) + '\n Mètode de protecció: ' + str(path.split("/")[-1]) + " amb k = 2"
            else:
                xLabel = "k"
            ax.set_xlabel(xLabel)
            ax.set_ylabel('Percentatge de similaritat (%)')
            ax.legend()
            ax.grid(True)
            ax.set_title(title)
            
            ax.set_ylim([-5, 105])
            ax.set_yticks(np.arange(0, 101, 20))

        plt.show()

    def viewMeanSimilaritiesGrouped(self, files, name):
        folders = dp.METRICS

        n_rows = len(files)
        n_cols = len(folders)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.0), squeeze=False)
        fig.suptitle(f"Mitjana de mètriques de similaritat en {name}", fontsize=12)
        fig.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.15)  # espacio inferior para leyenda

        # Lista para almacenar todos los manejadores de leyendas
        all_legend_handles = []
        all_legend_labels = []

        for row_idx, file in enumerate(files):
            for col_idx, path in enumerate(folders):
                ax = axes[row_idx, col_idx]
                dir = os.path.join(path, file + ".json")

                try:
                    with open(dir, "r") as f:
                        data = json.load(f)
                except:
                    ax.set_visible(False)
                    continue

                for key in data:
                    listAverages = []
                    listParameters = []
                    for metric in data[key]:
                        if key != "Densities":
                            meanMetric = np.array(metric[0]).mean()
                            parameter = metric[1]
                            listAverages.append(meanMetric)
                            listParameters.append(parameter)

                    if not listAverages:
                        continue

                    # Ordenar las métricas por parámetros
                    listParameters, listAverages = zip(*sorted(zip(listParameters, listAverages)))

                    # Graficar la línea y añadirla a la lista de leyendas
                    line, = ax.plot(listParameters, listAverages, marker='o', label=key)
                    if key not in all_legend_labels:
                        all_legend_handles.append(line)
                        all_legend_labels.append(key)

                method = path.split("/")[-1]
                ax.set_title(f"{file} - {method}", fontsize=8)

                # Solo mostrar etiquetas en los ejes exteriores
                xLabel = r'$\epsilon$' if method == "ELDP" else "k"
                if row_idx == n_rows - 1:
                    ax.set_xlabel(xLabel, fontsize=8)
                else:
                    ax.set_xticklabels([])  # No mostrar etiquetas de x
                    ax.set_xlabel("")  # Sin título en el eje x

                if col_idx == 0:
                    ax.set_ylabel("Similaritat (%)", fontsize=8)
                else:
                    ax.set_yticklabels([])  # No mostrar etiquetas de y
                    ax.set_ylabel("")  # Sin título en el eje y

                ax.set_ylim([-5, 105])
                ax.set_yticks(np.arange(0, 101, 20))
                ax.grid(True)
                ax.tick_params(axis='both', labelsize=7)

        # Leyenda global (abajo centrada)
        if all_legend_handles:
            fig.legend(all_legend_handles, all_legend_labels, loc='lower center', ncol=len(all_legend_labels), fontsize='medium')

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # espacio para leyenda y título
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

    def viewDensities(self, file):
        """Visualitzar les densitats dels grafs originals i protegits"""
        folders = dp.METRICS

        for path in folders:
            dir = path + "/" + file + ".json"
            try:
                with open(dir, "r") as f:
                    data = json.load(f)
            except:
                continue

            # Ordenar los resultados por el valor del parámetro
            sorted_results = sorted(data["Densities"], key=lambda x: x[1])
            num_plots = len(sorted_results)

            if num_plots == 1:
                # Caso especial: solo un gráfico → fig y ax simples
                fig, ax = plt.subplots(figsize=(6, 4))
                result = sorted_results[0]
                parameter = result[1]
                listOriginal = [d[0] for d in result[0]]
                listProtected = [d[1] for d in result[0]]
                listGraphs = list(range(len(result[0])))

                ax.plot(listGraphs, listOriginal, marker='o', label="Graf original")
                ax.plot(listGraphs, listProtected, marker='o', label="Graf protegit")

                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Densitat")

                typeProtection = r'$\epsilon$' if path.split("/")[-1] == "ELDP" else "k"
                ax.set_title(f"{typeProtection} = {parameter}")
                ax.grid(True)

                fig.suptitle(f'Densitats en {FILE}\nMètode de protecció: {path.split("/")[-1]}', fontsize=12)
                fig.legend(loc='lower center', ncol=2)
                fig.subplots_adjust(top=0.85, bottom=0.15)

            else:
                # Múltiples plots → usar subplots organizados
                cols = 3
                rows = (num_plots + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
                fig.suptitle(f'Densitats en {FILE}\nMètode de protecció: {path.split("/")[-1]}', fontsize=12)
                fig.subplots_adjust(hspace=0.6, wspace=0.4, top=0.88, bottom=0.08)

                for idx, result in enumerate(sorted_results):
                    row, col = divmod(idx, cols)
                    ax = axes[row][col]

                    listOriginal = [d[0] for d in result[0]]
                    listProtected = [d[1] for d in result[0]]
                    listGraphs = list(range(len(result[0])))
                    parameter = result[1]

                    ax.plot(listGraphs, listOriginal, marker='o', label="Graf original")
                    ax.plot(listGraphs, listProtected, marker='o', label="Graf protegit")

                    if row == rows - 1:
                        ax.set_xlabel("Timestamp")
                    if col == 0:
                        ax.set_ylabel("Densitat")

                    typeProtection = r'$\epsilon$' if path.split("/")[-1] == "ELDP" else "k"
                    ax.set_title(f"{typeProtection} = {parameter}")
                    ax.grid(True)

                # Eliminar subplots vacíos
                for i in range(num_plots, rows * cols):
                    row, col = divmod(i, cols)
                    fig.delaxes(axes[row][col])

                # Añadir una única leyenda común
                handles, labels = axes[0][0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=2)

            plt.show()

    def viewDegrees(self, file):
        """Visualitzar les densitats dels grafs originals i protegits"""
        folders = dp.METRICS

        for path in folders:
            dir = path + "/" + file + ".json"
            try:
                with open(dir, "r") as f:
                    data = json.load(f)
            except:
                continue

            # Ordenar los resultados por el valor del parámetro
            sorted_results = sorted(data["Degrees"], key=lambda x: x[1])
            num_plots = len(sorted_results)

            if num_plots == 1:
                # Caso especial: solo un gráfico → fig y ax simples
                fig, ax = plt.subplots(figsize=(6, 4))
                result = sorted_results[0]
                parameter = result[1]
                meanOriginal = [d[0][0] for d in result[0]]
                medianOriginal = [d[0][1] for d in result[0]]
                meanProtected = [d[1][0] for d in result[0]]
                medianProtected = [d[1][1] for d in result[0]]
                listGraphs = list(range(len(result[0])))

                ax.plot(listGraphs, meanOriginal, marker='o', label="Mitjana graus original", color = '#1f77b4')
                ax.plot(listGraphs, medianOriginal, marker='x', label="Mediana graus original", color = '#1f77b4', linestyle='--')
                ax.plot(listGraphs, meanProtected, marker='o', label="Mitjana graus protegit", color = '#ff7f0e')
                ax.plot(listGraphs, medianProtected, marker='x', label="Mediana graus protegit", color = '#ff7f0e', linestyle='--')


                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Resultat")

                typeProtection = r'$\epsilon$' if path.split("/")[-1] == "ELDP" else "k"
                ax.set_title(f"{typeProtection} = {parameter}")
                ax.grid(True)

                fig.suptitle(f'Mètriques de graus en {FILE}\nMètode de protecció: {path.split("/")[-1]}', fontsize=12)
                fig.legend(loc='lower center', ncol=2)
                fig.subplots_adjust(top=0.85, bottom=0.15)

            else:
                # Múltiples plots → usar subplots organizados
                cols = 3
                rows = (num_plots + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
                fig.suptitle(f'Mètriques de graus en {FILE}\nMètode de protecció: {path.split("/")[-1]}', fontsize=12)
                fig.subplots_adjust(hspace=0.6, wspace=0.4, top=0.88, bottom=0.08)

                for idx, result in enumerate(sorted_results):
                    row, col = divmod(idx, cols)
                    ax = axes[row][col]

                    meanOriginal = [d[0][0] for d in result[0]]
                    medianOriginal = [d[0][1] for d in result[0]]
                    meanProtected = [d[1][0] for d in result[0]]
                    medianProtected = [d[1][1] for d in result[0]]

                    
                    listProtected = [d[1] for d in result[0]]
                    listGraphs = list(range(len(result[0])))
                    parameter = result[1]

                    ax.plot(listGraphs, meanOriginal, marker='o', label="Mitjana graus original", color = '#1f77b4')
                    ax.plot(listGraphs, medianOriginal, marker='x', label="Mediana graus original", color = '#1f77b4', linestyle='--')
                    ax.plot(listGraphs, meanProtected, marker='o', label="Mitjana graus protegit", color = '#ff7f0e')
                    ax.plot(listGraphs, medianProtected, marker='x', label="Mediana graus protegit", color = '#ff7f0e', linestyle='--')

                    if row == rows - 1:
                        ax.set_xlabel("Timestamp")
                    if col == 0:
                        ax.set_ylabel("Resultat")

                    typeProtection = r'$\epsilon$' if path.split("/")[-1] == "ELDP" else "k"
                    ax.set_title(f"{typeProtection} = {parameter}")
                    ax.grid(True)

                # Eliminar subplots vacíos
                for i in range(num_plots, rows * cols):
                    row, col = divmod(i, cols)
                    fig.delaxes(axes[row][col])

                # Añadir una única leyenda común
                handles, labels = axes[0][0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=2)

            plt.show()

    def visualizeMetrics(self):
        """Visualitza totes les mètriques generades d'un fitxer.
        """
        if FILE.endswith("CollegeMsg"):
            files = ["HOUR_CollegeMsg", "DAY_CollegeMsg", "WEEK_CollegeMsg", "MONTH_CollegeMsg"]
            name = "CollegeMsg"
        else:
            files = ["HOUR_ia-enron-employees", "DAY_ia-enron-employees", "WEEK_ia-enron-employees", "MONTH_ia-enron-employees"]
            name  = "ia-enron-employees"

        if  FILE.endswith("CollegeMsg") or FILE.endswith("ia-enron-employees"):
            self.viewMeanSimilaritiesGrouped(files, name)
            for file in files:
                self.viewDensities(file)
        else:
            self.viewMeanSimilarities()
            self.viewIndividualSimilarities()
            self.viewDensities(FILE)
            self.viewDegrees(FILE)


if __name__ == "__main__":
    metric = Metrics(mode = 2)