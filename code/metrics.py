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
         'mammalia-voles-rob-trapping',
         'aves-sparrow-social'] 

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
        """Llegir les mètriques dels fitxers originals i protegits

        Returns:
            Dict, Dict: Diccionari amb els fitxers originals i protegits.
        """
        folders = dp.OUTPUTS
        originalFiles = {}
        protectedDict = {}

        for root_folder in folders:
            method = os.path.basename(root_folder)
            if method == "original_graphs":
                for dirpath, _, filenames in os.walk(root_folder):
                    for f in filenames:
                        nameFile = f.split(".")[0]
                        originalFiles[nameFile] = (dirpath, f)
                continue

            for dirpath, _, filenames in os.walk(root_folder):
                for f in filenames:
                    if not f.endswith(".pkl"):
                        continue

                    nameFile = f.split(".")[0]
                    rel_path = os.path.relpath(dirpath, root_folder)  
                    full_method_key = method + "/" + rel_path          

                    if full_method_key not in protectedDict:
                        protectedDict[full_method_key] = {}

                    if nameFile not in protectedDict[full_method_key]:
                        protectedDict[full_method_key][nameFile] = []

                    try:
                        param = float(f.split("_")[-1].replace(".pkl", ""))
                    except ValueError:
                        param = -1  # valor por defecto

                    protectedDict[full_method_key][nameFile].append((dirpath, f, param))

        return originalFiles, protectedDict

    def computeMetrics(self):
        """Calcula les mètriques de tots els fitxers protegits i els guarda en JSON."""
        originalFiles, protectedDict = self.readMetrics()

        if not FILE:
            raise ValueError("No file selected. Please select a file to compute metrics.")

        with open(os.path.join(originalFiles[FILE][0], originalFiles[FILE][1]), 'rb') as f:
            originalGraphs = pickle.load(f)

        for method in protectedDict:
            if FILE not in protectedDict[method]:
                continue

            results = {
                "Jaccard": [], "DeltaConnectivity": [], "Jaccard Betweenness": [],
                "Jaccard Closeness": [], "Jaccard Degree": [], "Densities": [], "Degrees": []
            }

            for i in tqdm(protectedDict[method][FILE], desc="Calculant mètriques en: " + str(FILE) + " - " + method.split("/")[0] + method.split("/")[1][0]):
                listJaccard, listDeltaConnectivity = [], []
                listBetweeness, listCloseness, listDegreeCentrality = [], [], []
                listDensities, listDegrees = [], []

                with open(os.path.join(i[0], i[1]), 'rb') as f:
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
            output_path = os.path.join(current_dir, "metrics", method, FILE + ".json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(results, f)

    def viewMeanSimilarities(self):
        """Visualitzar les mètriques de similaritat de cada mètode en un gràfic, donat un fitxer"""
        folders = dp.METRICS
        results = {}

        # Pas 1: Recollir totes les mètriques de cada mètode
        for path in folders:
            method = os.path.basename(path)
            if method not in results:
                results[method] = {}

            for root, dirs, files in os.walk(path):
                for file in files:
                    if not file.endswith(".json") or not file.startswith(FILE):
                        continue

                    with open(os.path.join(root, file), "r") as f:
                        data = json.load(f)

                    for metric, entries in data.items():
                        if metric in ["Densities", "Degrees"]:
                            continue

                        if metric not in results[method]:
                            results[method][metric] = {}


                        for values, param in entries:
                            values = np.array(values)
                            param = float(param)
                            if param not in results[method][metric]:
                                results[method][metric][param] = []
                            results[method][metric][param].append(values.mean())

        # En cas d'aquest fitxer que no s'ha fet ELDP, només mostrar KDA.
        if FILE.startswith("insecta-ant"):
            results = {"KDA": results["KDA"]}

        # Pas 2: Graficar
        num_methods = len(results)
        fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 5), squeeze=False)
        fig.tight_layout(pad=5.0)

        handles_labels = []

        for idx, (method, metrics) in enumerate(results.items()):
            row, col = divmod(idx, 3)
            ax = axes[row][col]
            
            for metric, param_dict in metrics.items():
                params = sorted(param_dict.keys())
                means = [np.mean(param_dict[p]) for p in params]
                stds = [np.std(param_dict[p]) for p in params]

                line = ax.plot(params, means, label=metric, marker='o')[0]
                ax.fill_between(params,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.2)

            title = f'Mitjana de mètriques en {FILE}\nMètode: {method}, #Experiments = 5'
            xLabel = r'$\epsilon$' if method == "ELDP" else "k"

            ax.set_title(title)
            ax.set_xlabel(xLabel)
            ax.set_ylabel("Similaritat (%)")
            ax.set_ylim([-5, 105])
            ax.set_yticks(np.arange(0, 101, 10))
            ax.grid(True)
            ax.set_axisbelow(True)

            h, l = ax.get_legend_handles_labels()
            handles_labels.extend(zip(h, l))

        # Llegendes
        seen = set()
        unique_handles_labels = []
        for h, l in handles_labels:
            if l not in seen:
                unique_handles_labels.append((h, l))
                seen.add(l)

        handles, labels = zip(*unique_handles_labels)
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.02))
        plt.subplots_adjust(bottom=0.2)

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
        base_folders = dp.METRICS  # Ej. ['metrics/ELDP', 'metrics/KDA']
        all_metrics = ["Jaccard", "DeltaConnectivity", "Jaccard Betweenness", "Jaccard Closeness", "Jaccard Degree"]
        num_methods = len(base_folders)

        cols = num_methods
        if FILE.startswith("insecta-ant"):
            cols = 1
            
        rows = len(all_metrics) * int(np.ceil(num_methods / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5), squeeze=False)
        fig.suptitle(f"Mitjana de mètriques de similaritat per timestamp en {FILE} \n #Experiments = 5", fontsize=12)
        fig.subplots_adjust(hspace=0.4, wspace=0.1, top=0.95, right=0.9)

        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
        show_cbar = True

        for metric_idx, metric_name in enumerate(all_metrics):
            for method_idx, base_path in enumerate(base_folders):
                row = metric_idx * int(np.ceil(num_methods / cols)) + method_idx % int(np.ceil(num_methods / cols))
                col = method_idx // int(np.ceil(num_methods / cols))
                ax = axes[row][col]

                all_experiment_values = {}
                all_param_labels = set()

                for subfolder in sorted(os.listdir(base_path)):
                    subfolder_path = os.path.join(base_path, subfolder)
                    if not os.path.isdir(subfolder_path):
                        continue

                    # Buscar recursivamente archivos JSON cuyo nombre empiece por FILE
                    for root, _, files in os.walk(subfolder_path):
                        for file in files:
                            if file.endswith(".json") and file.startswith(FILE):
                                json_path = os.path.join(root, file)
                                try:
                                    with open(json_path, "r") as f:
                                        data = json.load(f)
                                except Exception:
                                    continue

                                if metric_name not in data:
                                    continue

                                for metric in data[metric_name]:
                                    values = metric[0]
                                    label = metric[1]
                                    all_param_labels.add(label)

                                    if label not in all_experiment_values:
                                        all_experiment_values[label] = []
                                    all_experiment_values[label].append(values)

                if not all_experiment_values:
                    ax.set_visible(False)
                    continue

                mean_values_matrix = []
                sorted_labels = sorted(all_param_labels)
                for label in sorted_labels:
                    group = np.array(all_experiment_values[label])
                    mean = np.mean(group, axis=0)
                    mean_values_matrix.append(mean)

                df = pd.DataFrame(mean_values_matrix, index=sorted_labels).T

                sns.heatmap(df, ax=ax, cbar=show_cbar, cbar_ax=cbar_ax if show_cbar else None,
                            vmin=0, vmax=100, linewidths=0.5, linecolor='gray')
                show_cbar = False

                ax.set_title(f"{metric_name} - {base_path.split('/')[-1]}", fontsize=9)

                if col == 0:
                    ax.set_ylabel("Timestamp", fontsize=8)
                else:
                    ax.set_ylabel("")
                    ax.set_yticks([])

                if row == rows - 1:
                    xLabel = r'$\epsilon$' if "ELDP" in base_path else "k"
                    ax.set_xlabel(xLabel, fontsize=8)
                else:
                    ax.set_xlabel("")
                    ax.set_xticks([])

                ax.tick_params(axis='both', labelsize=7)

        cbar_ax.set_ylabel("Similaritat (%)", fontsize=10)
        cbar_ax.tick_params(labelsize=8)
        plt.show()

    def viewDensities(self, file):
        """Visualitzar les densitats per cada paràmetre utilitzat de cada algorisme"""
        folders = dp.METRICS  

        for path in folders:
            all_results = []

            # Buscar recursivamente archivos .json que empiecen por el nombre del fichero
            for subfolder in sorted(os.listdir(path)):
                subfolder_path = os.path.join(path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                for root, _, files in os.walk(subfolder_path):
                    for fname in files:
                        if fname.endswith(".json") and fname.startswith(file):
                            json_path = os.path.join(root, fname)
                            try:
                                with open(json_path, "r") as f:
                                    data = json.load(f)
                                    all_results.extend(data.get("Densities", []))
                            except:
                                continue

            if not all_results:
                continue

            # Agrupar por valor del parámetro (ε o k)
            grouped_results = {}
            for result in all_results:
                densities = result[0]  # lista de pares (original, protegido)
                parameter = result[1]

                if parameter not in grouped_results:
                    grouped_results[parameter] = []

                grouped_results[parameter].append(densities)

            sorted_keys = sorted(grouped_results.keys())
            num_plots = len(sorted_keys)

            # Crear subplots
            cols = 3
            rows = (num_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
            fig.suptitle(f'Densitats en {file}\nMètode de protecció: {path.split("/")[-1]}, #Experiments = 5', fontsize=12)
            fig.subplots_adjust(hspace=0.6, wspace=0.4, top=0.88, bottom=0.08)

            for idx, key in enumerate(sorted_keys):
                row, col = divmod(idx, cols)
                ax = axes[row][col]
                group = grouped_results[key]

                # Transponer para obtener listas separadas de densitats originals i protegides
                original_all = np.array([[x[0] for x in run] for run in group])
                protected_all = np.array([[x[1] for x in run] for run in group])

                timestamps = np.arange(original_all.shape[1])

                original_mean = np.mean(original_all, axis=0)
                protected_mean = np.mean(protected_all, axis=0)

                original_std = np.std(original_all, axis=0)
                protected_std = np.std(protected_all, axis=0)

                # Línea + desviació típica
                ax.plot(timestamps, original_mean, label="Graf original", marker='o')
                ax.fill_between(timestamps, original_mean - original_std, original_mean + original_std, alpha=0.2)

                ax.plot(timestamps, protected_mean, label="Graf protegit", marker='o')
                ax.fill_between(timestamps, protected_mean - protected_std, protected_mean + protected_std, alpha=0.2)

                if row == rows - 1:
                    ax.set_xlabel("Timestamp")
                if col == 0:
                    ax.set_ylabel("Densitat")

                typeProtection = r'$\epsilon$' if "ELDP" in path else "k"
                ax.set_title(f"{typeProtection} = {key}")
                ax.grid(True)

            # Eliminar subplots buits
            for i in range(num_plots, rows * cols):
                row, col = divmod(i, cols)
                fig.delaxes(axes[row][col])

            # Llegenda comuna
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=2)

            plt.show()

    def viewDegrees(self, file):
        """Visualitzar les densitats dels grafs originals i protegits"""
        folders = dp.METRICS  

        for path in folders:
            all_results = []

            # Buscar recursivamente archivos .json que empiecen por el nombre del fichero
            for subfolder in sorted(os.listdir(path)):
                subfolder_path = os.path.join(path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                for root, _, files in os.walk(subfolder_path):
                    for fname in files:
                        if fname.endswith(".json") and fname.startswith(file):
                            json_path = os.path.join(root, fname)
                            try:
                                with open(json_path, "r") as f:
                                    data = json.load(f)
                                    all_results.extend(data.get("Degrees", []))
                            except:
                                continue

            if not all_results:
                continue

            # Agrupar por valor del parámetro (ε o k)
            grouped_results = {}
            for result in all_results:
                degrees = result[0]  # lista de pares (original, protegido)
                parameter = result[1]

                if parameter not in grouped_results:
                    grouped_results[parameter] = []

                grouped_results[parameter].append(degrees)

            sorted_keys = sorted(grouped_results.keys())
            num_plots = len(sorted_keys)

            # Crear subplots
            cols = 3
            rows = (num_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
            fig.suptitle(f'Graus en {file}\nMètode de protecció: {path.split("/")[-1]}, #Experiments = 5', fontsize=12)
            fig.subplots_adjust(hspace=0.2, wspace=0.2, top=0.88, bottom=0.125)

            for idx, key in enumerate(sorted_keys):
                row, col = divmod(idx, cols)
                ax = axes[row][col]
                group = grouped_results[key]

                # Transponer para obtener listas separadas de densitats originals i protegides
                original_means = np.array([[x[0][0] for x in run] for run in group])
                original_medians = np.array([[x[0][1] for x in run] for run in group])
                protected_means = np.array([[x[1][0] for x in run] for run in group])
                protected_medians = np.array([[x[1][1] for x in run] for run in group])
                
                timestamps = np.arange(original_means.shape[1])

                original_means_std = np.std(original_means, axis=0)
                original_medians_std = np.std(original_medians, axis=0)
                protected_means_std = np.std(protected_means, axis=0)
                protected_medians_std = np.std(protected_medians, axis=0)

                original_means = np.mean(original_means, axis=0)
                original_medians = np.mean(original_medians, axis=0)
                protected_means = np.mean(protected_means, axis=0)
                protected_medians = np.mean(protected_medians, axis=0)

                # Línea + desviació típica
                ax.plot(timestamps, original_means, label="Mitjana graus grafs originals", marker='o', color='#1f77b4')
                ax.fill_between(timestamps, original_means - original_means_std, original_means + original_means_std, alpha=0.2, color='#1f77b4')
                
                ax.plot(timestamps, original_medians, label="Mediana graus grafs originals", marker='x', color='#1f77b4', linestyle='--')
                ax.fill_between(timestamps, original_medians - original_medians_std, original_medians + original_medians_std, alpha=0.2, color='#1f77b4')

                ax.plot(timestamps, protected_means, label="Mitjana graus grafs protegits", marker='o', color='#ff7f0e')
                ax.fill_between(timestamps, protected_means - protected_means_std, protected_means + protected_means_std, alpha=0.2, color='#ff7f0e')
                
                ax.plot(timestamps, protected_medians, label="Mediana graus grafs protegits", marker='x', linestyle='--', color='#ff7f0e')
                ax.fill_between(timestamps, protected_medians - protected_medians_std, protected_medians + protected_medians_std, alpha=0.2, color='#ff7f0e')   

                if row == rows - 1:
                    ax.set_xlabel("Timestamp")
                if col == 0:
                    ax.set_ylabel("Graus")

                typeProtection = r'$\epsilon$' if "ELDP" in path else "k"
                ax.set_title(f"{typeProtection} = {key}")
                ax.grid(True)

            # Eliminar subplots buits
            for i in range(num_plots, rows * cols):
                row, col = divmod(i, cols)
                fig.delaxes(axes[row][col])

            # Llegenda comuna
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
                self.viewDegrees(file)
        else:
            self.viewMeanSimilarities()
            self.viewIndividualSimilarities()
            self.viewDensities(FILE)
            self.viewDegrees(FILE)


if __name__ == "__main__":
    metric = Metrics(mode = 1)