import networkx as nx
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
from collections import Counter

# import pandas as pd
# import cv2
# import tkinter as tk

# Obtenir resolució de la pantalla
# root = tk.Tk()
# WIDTH = root.winfo_screenwidth()
# HEIGHT = root.winfo_screenheight()
# root.destroy()

class GraphProtection:  
    """Mòdul creador de grafs
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('path', 'weighted', 'directed', 'format', 'df',
                 'grouped_df', 'graph', 'node_positions', 'filename')
    def __init__(self, filename, input_tuple, df):
        """Inicialització de la classe

        Args:
            filename (str): Nom del arxiu
            input_tuple (tuple): Tuple amb la informació del dataset (path, weighted, directed, format)
            df (pd.DataFrame): Dataframe amb la informació del dataset
        """
        self.path, self.weighted, self.directed, self.format = input_tuple
        self.df = df
        self.grouped_df = self.get_grouped_df()
        self.graph = self.create_graph()
        self.node_positions = nx.spectral_layout(self.graph) # Posicions fixes dels nodes del graf
        self.filename = filename

    def create_graph(self):
        """Creació de tots els nodes del graf.  

        Returns:
            nx.graph: Graf generat sense cap relació entre nodes
        """
        nodes = set(self.df["From"]).union(set(self.df["To"])) # Fem l'unió de tots els nodes únics del dataset
        if self.directed:
            graph = nx.DiGraph() 
        else:
            graph = nx.Graph() # Creem un graf amb networkx, dirigit o no segons el paràmetre directed 
        
        graph.add_nodes_from(nodes) # Afegim els nodes en el graf
        
        return graph

    def get_grouped_df(self):
        grouped_df = self.df.groupby("Timestamp")
        return grouped_df

    def iterate_graph(self, group):
        self.graph.clear_edges() # Netejem les arestes del anterior plot
        if not self.weighted:
            self.graph.add_edges_from(zip(group["From"], group["To"])) # Afegim les arestes del timestamp o data actual
        else:
            edges = zip(group["From"], group["To"], group["Weight"])
            self.graph.add_weighted_edges_from(edges)

    def visualize_graph(self):
        """Visualitzar cada timestamp del graf temporal
        """
        plt.figure(figsize=(30, 30)) # Tamany de la figura 
        for time, group in self.grouped_df:
            self.iterate_graph(group)
            plt.clf()  # Netejem l'anterior plot
            plt.title(f"Plotting {self.filename} temporal graph") # Afegim el títol del gràfic
            plt.text(0.51, 0.88, f"Exchanges of messages at time: {time}", ha = "center", va="top", transform=plt.gcf().transFigure) # Afegim un subtítol 
            # Dibuixem el graf en les posicions corresponents
            nx.draw(self.graph, self.node_positions, with_labels=True, node_color='lightblue', edge_color='red', node_size=50, font_size=3, arrows=True, width=0.5)
            plt.pause(0.1)  # Pausar per visualitzar canvis 
        plt.show() # Mostrar el graf
        
    def save_graphs(self, original_graphs, protected_graphs, algorithm, parameter):
        og = "code/output/original_graphs/" + str(self.filename) + ".pkl"
        pg = "code/output/" + str(algorithm) + "/" + str(self.filename) + "_" + str(parameter) + ".pkl"

        with open(og, "wb") as f:
            pickle.dump(original_graphs, f)

        with open(pg, "wb") as f:
            pickle.dump(protected_graphs, f)
            
    # def create_animation(self, grouped_df):
    #     output = "CollegeMsg.mp4"
    #     frame_rate = 2  # Adjust speed

    #     # Set high DPI and figure size
    #     dpi = 200  # High DPI for better quality
    #     fig_width = WIDTH / dpi
    #     fig_height = HEIGHT / dpi

    #     fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    #     plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)  # Avoid title cutoff

    #     # Get figure size in pixels
    #     fig.canvas.draw()
    #     width, height = fig.canvas.get_width_height()

    #     # Define OpenCV video writer
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     video = cv2.VideoWriter(output, fourcc, frame_rate, (width, height))

    #     for time, group in grouped_df:
    #         print(f"Actual date time: {time}")
    #         self.graph.clear_edges()
    #         self.graph.add_edges_from(zip(group["From"], group["To"]))

    #         ax.clear()
    #         ax.set_title(f"Missatges enviats en la data: {time}", fontsize=12.5, fontweight='bold', pad=20)

    #         # Draw graph with better sizing
    #         nx.draw(self.graph, self.node_positions, with_labels=True, node_color='lightblue', edge_color='red', node_size=40, font_size=2, arrows=True, width=0.5)

    #         # Convert Matplotlib figure to OpenCV frame
    #         fig.canvas.draw()
    #         frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #         frame = frame.reshape(height, width, 3)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    #         video.write(frame)
            
    #     video.release()
    #     print("Video saved successfully!")

    # def animate_graph(self):
    #     self.df["Date"] = pd.to_datetime(self.df["Timestamp"], unit="s").dt.date
    #     grouped_df = self.df.groupby("Date")
    #     self.create_animation(grouped_df)

    # def visualize_per_day(self):
    #     """Visualitzar el graf temporal agrupat per dies 
    #     """
    #     self.df["Date"] = pd.to_datetime(self.df["Timestamp"], unit="s").dt.date
    #     grouped_df = self.df.groupby("Date")
    #     self.visualize_graph(grouped_df)

class ELDP(GraphProtection):
    __slots__ = ('nodes', 'epsilon')
    def __init__(self, filename, input_tuple, df, epsilon=0.5):
        super().__init__(filename, input_tuple, df)
        self.epsilon = epsilon
        self.nodes = self.graph.number_of_nodes()

    def compute_density(self):
        """Calcular la densitat mitjana de tots els grafs que conforma un dataset

        Returns:
            float: Densitat mitjana
        """
        density = nx.density(self.graph)
        return density
    
    def compute_probabilities(self, density):
        """Calcular les probabilitats que s'usaràn per fer els noise-graphs

        Args:
            epsilon (float): Paràmetre Epsilon Local Edge Differential Privacy. Segons el seu valor, els grafs tindràn més o menys soroll.

        Returns:
            p0, p1: Probabilitats d'afegir o treure arestes pels noise-graphs
        """
        p0 = 1 - (1 / ((math.exp(self.epsilon) - 1 + (1 / density))))
        p1 = (math.exp(self.epsilon)) / ((math.exp(self.epsilon) - 1 + (1 / density)))
        return p0, p1

    def complement_graph(self):
        """Crear el complementari d'un graf 

        Returns:
            nx.graph: Graf complementari
        """
        graph = nx.complement(self.graph)
        return graph

    def gilbert_graph(self, p):
        """Crear un graf de soroll 

        Args:
            p (float): Probabilitat d'afegir una aresta 

        Returns:
            nx.graph: Noise graph 
        """
        # Depenent de si el graf original és dirigit o no, fem el graf de soroll erdos_renyi amb direccions o sense
        graph = nx.erdos_renyi_graph(self.nodes, p, directed=self.directed)
        return graph

    def intersection_graph(self, g1, g2):
        """Realitzar l'intersecció d'arestes entre grafs

        Args:
            g1, g2 (nx.graph): Graf a comparar arestes

        Returns:
            nx.graph: Intersecció de grafs
        """
        # Realitzem l'intersecció de dos grafs
        graf = nx.intersection(g1, g2)
        return graf
    
    def xor_graph(self, g1, g2):
        """Suma de grafs (operació XOR)

        Args:
            g1, g2 (nx.graph): Grafs a sumar

        Returns:
            nx.graph: Graf sumat
        """
        # Realitzem la suma XOR, fent la diferència simètrica
        graf = nx.symmetric_difference(g1, g2)
        return graf
        
    def apply_protection(self):
        """Aplicar protecció l-EDP en el dataset

        Args:
            p0, p1 (float): Probabilitats pels noise-graphs

        Returns:
            List[nx.graph], List[nx.graph]: Llista de datasets originals i datasets protegits
        """
        original_graphs = list()
        protected_graphs = list()

        for _, group in self.grouped_df:
            # Iterem a la següent snapshot
            self.iterate_graph(group)
            # Calculem el seu graf complementari
            original_graphs.append(self.graph.copy())
            complement_graph = self.complement_graph()

            # Mirem la densitat del graf. En cas de ser major a 0.5 no es pot realitzar ε-ELDP
            density = self.compute_density()
            assert density <= 0.5, "No es pot protegir el graf temporal, degut a que és massa dens per complir ε-ELDP"

            # Calculem les probabilitats d'afegir i treure arestes, i computem els noise graphs
            p0,p1 = self.compute_probabilities(density)
            noise_g0 = self.gilbert_graph(1-p0)
            noise_g1 = self.gilbert_graph(1-p1)

            # Intersecció dels sorolls amb el graf original i el seu complementari
            g0 = self.intersection_graph(noise_g0, complement_graph) 
            g1 = self.intersection_graph(noise_g1, self.graph)

            # Realitzem un XOR del graf original, amb els grafs de sorolls computats
            sum1 = self.xor_graph(self.graph, g0)
            protected_g = self.xor_graph(sum1, g1)
            protected_graphs.append(protected_g)

        return original_graphs, protected_graphs
             
class KDA(GraphProtection):
    __slots__ = ('k', 'm', 'T', 'degreeMatrix', 'indegreeMatrix', 'outdegreeMatrix')
    def __init__(self, filename, input_tuple, df, k=2):
        super().__init__(filename, input_tuple, df)
        self.k = k
        self.m = math.ceil(self.graph.number_of_nodes() / self.k)
        self.T = len(self.grouped_df)
        self.degreeMatrix, self.indegreeMatrix, self.outdegreeMatrix = self.get_degreeMatrix()

    def get_degree(self, type):
        """Obtenir els graus de tots els nodes d'un graf, incloent els de grau 0.

        Args:
            type (str): Tipus de grau (in_degree, out_degree, degree)

        Returns:
            List: Llista de graus del graf, assegurant que tots els nodes hi són.
        """
        if type == "outdegree":
            degree_dict = dict(self.graph.out_degree())
        elif type == "indegree":
            degree_dict = dict(self.graph.in_degree())
        else:
            degree_dict = dict(self.graph.degree())

        # Retornar els graus assegurant que surtin tots encara que sigui 0
        return [degree_dict.get(node, 0) for node in self.graph.nodes()]

    def get_degreeMatrix(self):
        """Obtenir les matrius de graus dels grafs. Cada fila correspon els graus d'un graf

        Returns:
            np.array(), np.array(), np.array(): Matrius degree, in_degree i out_degree
        """
        degreeMatrix = None
        indegreeMatrix = None
        outdegreeMatrix = None

        # En cas de ser un graf dirigit, obtenir les in_degree i out_degree matrices
        if self.directed:
            for _, group in self.grouped_df:
                self.iterate_graph(group)
                indegree_array = self.get_degree("indegree")
                outdegree_array = self.get_degree("outdegree")

                if indegreeMatrix is None:
                    indegreeMatrix = indegree_array 
                else:
                    indegreeMatrix = np.vstack((indegreeMatrix, indegree_array))

                if outdegreeMatrix is None:
                    outdegreeMatrix = outdegree_array
                else:
                    outdegreeMatrix = np.vstack((outdegreeMatrix, outdegree_array))

            return None, indegreeMatrix, outdegreeMatrix

        # En cas de ser no dirigit, fer només una matriu de graus
        for _, group in self.grouped_df:
            self.iterate_graph(group)
            degree_array = self.get_degree("degree")
            if degreeMatrix is None:
                degreeMatrix = degree_array
            else:
                degreeMatrix = np.vstack((degreeMatrix, degree_array))

        return degreeMatrix, None, None

    def compute_PMatrix(self, degreeMatrix):
        """Calcular matriu de medianes P 

        Args:
            degreeMatrix (np.array): Matriu de graus a particionar

        Returns:
            np.array() : Matriu de medianes 
        """
        aux_matrix = degreeMatrix.copy()
        PMatrix = np.zeros((self.T, self.m), dtype=int)
        for i, d_seq in enumerate(aux_matrix):
            # Aleatoritzem la seqüència de graus
            np.random.shuffle(d_seq)
            # Particionem la seqüència en m particions
            particions = np.array_split(d_seq, self.m)
            # Calculem la mediana de cada partició, i l'incorporem a la matriu final
            PMatrix[i] = np.array([int(round(np.median(p))) for p in particions])
        return PMatrix

    def anonymizeDegrees(self, degreeMatrix, PMatrix):
        """K-Anonimitzar graus de cada seqüència de la matriu degreeMatrix

        Args:
            degreeMatrix (np.array): Matriu de graus
            PMatrix (np.array): Matriu de medianes 

        Returns:
            np.array(): Matriu de graus anonimitzat
        """
        # Paràmetres necessaris 
        T, n = degreeMatrix.shape

        anonymizedDegrees = np.zeros_like(degreeMatrix, dtype=float)

        for t in range(T):
            # Ordenar els graus per agrupar-los correctament 
            sortedIndices = np.argsort(degreeMatrix[t])
            sortedDegrees = degreeMatrix[t][sortedIndices]
            auxmedian = None

            # Assignar la mediana a cada conjunt de particions
            for i in range(self.m):
                initial_idx = i*self.k # Comencem pel índex 0, en la següent iteració k*i
                final_idx = initial_idx + self.k # Comencem per l'index inicial + k, així succesivament 

                #En cas de que es trobin dintre del rang els índexos, s'assigna la mediana al conjunt de nodes
                if final_idx <= n: 
                    median = PMatrix[t,i]
                    sortedDegrees[initial_idx:final_idx] = median
                    auxmedian = median 
                # En cas contrari, utilitzem l'anterior mediana per tal d'assegurar k-anonymity en tota la seqüència
                else:
                    sortedDegrees[initial_idx:n] = auxmedian

            anonymizedDegrees[t, sortedIndices] = sortedDegrees
        
        return anonymizedDegrees
    
    def isEvenSequence(self, degreeSequence):
        """Comprovar que una seqüència de graus, el seu sumatori és parell

        Args:
            degreeSequence (np.array): Seqüència de graus d'un graf

        Returns:
            bool: El sumatori de la seqüència és parell o no
        """
        return sum(degreeSequence) % 2 == 0
    
    def isRealizableSequence(self, degreeSequence):
        """Comprovar si una seqüència de graus és realizable

        Args:
            degreeSequence (np.array): Seqüència de graus d'un graf

        Returns:
            bool: El graf és o no és realizable
        """
        degreeSequence = sorted(degreeSequence, reverse=True)
        # Condició 1. La seqüència és parell 
        if self.isEvenSequence(degreeSequence):
            # Condició 2. Compleix Erdős-Gallai 
            n = len(degreeSequence)
            cumulativeSum = np.cumsum(degreeSequence)
            for k in range(1, n + 1):
                rightSum = k * (k-1) + sum(min(d, k) for d in degreeSequence[k:])
                if cumulativeSum[k-1] > rightSum:
                    return False
            return True
        return False
    
    def splitSequences(self, degreeMatrix):
        """Obtenir diccionari de les seqüències realizables i no realizables

        Args:
            degreeMatrix (np.array): Matriu de graus 

        Returns:
            dictRealizable, dictNorealizable: Diccionaris amb clau -> seqüència (Tuple), 
                                                            i valor -> llista d'índexos de la matriu (List)
        """
        dictRealizable = {}
        dictNorealizable = {}

        for idx, seq in enumerate(degreeMatrix):
            tupleSequence = tuple(seq)
            if self.isRealizableSequence(seq):
                if tupleSequence not in dictRealizable:
                    dictRealizable[tupleSequence] = []
                dictRealizable[tupleSequence].append(idx)
            else:
                if tupleSequence not in dictNorealizable:
                    dictNorealizable[tupleSequence] = []
                dictNorealizable[tupleSequence].append(idx)

        return dictRealizable, dictNorealizable

    def compute_l1Distance(self, sequence1, sequence2):
        """Computar la l1 distance entre dos seqüències de graus

        Args:
            sequence1 (np.array): Primera seqüència de graus
            sequence2 (np.array): Segona seqüència de graus

        Returns:
            float: Distància de les dos seqüències
        """
        return np.sum(np.abs(np.array(sequence1) - np.array(sequence2)))

    def resolveNoRealizables(self, degreeMatrix, dictRealizable, dictNorealizable):
        """Trobar la millor opció per les seqüències no realizable

        Args:
            degreeMatrix (np.array): Matriu de graus Txn
            dictRealizable (Dict): Diccionari de seqüències realizables, amb l'índex de la matriu de graus on es troba
            dictNorealizable (Dict):  Diccionari de seqüències no realizables, amb l'índex de la matriu de graus on es troba
        """
        # Iterem totes les seqüències no realizables
        for k1, v1 in dictNorealizable.items():
            distances = {}
            # Iterem totes les seqüències realizables, i veiem quina és la que menys distància té
            for k2,v2 in dictRealizable.items():
                distances[k2] = self.compute_l1Distance(degreeMatrix[v1[0]], degreeMatrix[v2[0]]) 
            
            # Trobar la key de la seqüència realizable amb la mínima distància
            min_key = min(distances, key=distances.get)
            dictRealizable[min_key] += v1            

    def resolveNoRealizablesK(self, degreeMatrix, dictRealizable):
        """Trobar la millor opció per les seqüències realizables, que apareixen menys de k cops

        Args:
            degreeMatrix (np.array): Matriu de graus Txn
            dictRealizable (Dict): Diccionari de seqüències realizables, amb l'índex de la matriu de graus on es troba
        """
        minLength = min(len(lst) for lst in dictRealizable.values())
        min_key = [k for k, lst in dictRealizable.items() if len(lst) == minLength][0]

        while minLength < self.k:
            # Treiem el valor del diccionari
            value_removed = dictRealizable.pop(min_key)
            # Calculem les distàncies entre les altres opcions realizables
            distances = {}
            for key,value in dictRealizable.items():
                distances[key] = self.compute_l1Distance(degreeMatrix[value_removed[0]], degreeMatrix[value[0]]) 
            # Obtenim la distància mínima i l'afegim al diccionari 
            minimum = min(distances, key=distances.get)
            dictRealizable[minimum] += value_removed            

            # Següent iteració
            minLength = min(len(lst) for lst in dictRealizable.values())
            min_key = [k for k, lst in dictRealizable.items() if len(lst) == minLength][0]
            
    def realizeDegrees(self, degreeMatrix):
        """Modifcació de la matriu de graus 

        Args:
            degreeMatrix (np.array): Matriu de graus Txn

        Returns:
            np.array: Matriu de graus canviada perquè es puguin dibuixar els grafs 
        """
        dictRealizable, dictNorealizable = self.splitSequences(degreeMatrix)
        # print(f"dictRealizable: {dictRealizable.values()}")
        # print(f"dictNorealizable: {dictNorealizable.values()}")
        # print(f"File: {self.filename}, K = {self.k}")
        self.resolveNoRealizables(degreeMatrix, dictRealizable, dictNorealizable)
        self.resolveNoRealizablesK(degreeMatrix, dictRealizable)
        # print(f"dictRealizable: {dictRealizable}")

        # Fer remplaç de valors en la matriu de graus
        finalMatrix = degreeMatrix.copy()
        for new_values, indices in dictRealizable.items():
            finalMatrix[indices] = new_values

        finalMatrix = finalMatrix.astype(int)
        return finalMatrix

    def createProtectedGraphHavel(self, degreeSequence):
        """Crear un graf a partir d'una seqüència de nodes, amb la funció de networkx
           Característica: Com es reordenen els nodes, no sabem quin era l'identificació original

        Args:
            degreeSequence (np.array): Seqüència de graus

        Returns:
            nx.graph: Graf construït
        """
        degreeSequenceCopy = degreeSequence.copy()
        if self.directed:
            G = nx.havel_hakimi_graph(degreeSequenceCopy, create_using=nx.DiGraph)
        else:
            G = nx.havel_hakimi_graph(degreeSequenceCopy)
        return G
    
    def createProtectedGraphUndirected(self, degreeSequence):
        """Crear un graf a partir d'una seqüència de graus de forma manual
           Característica: Els nodes es mantenen en la seva posició original

        Args:
            degreeSequence (np.array): Seqüència de graus

        Returns:
            nx.graph: Graf construït
        """
        degreeSequenceCopy = degreeSequence.copy()
        n = len(degreeSequenceCopy) # Nombre de nodes
        nodes = list(range(n))  # Llista de nodes del graf
        degree_list = sorted(zip(degreeSequenceCopy, nodes), reverse=True)  # Ordenació del grau descendentment 
        
        G =nx.Graph() # Creem un no dirigit 
        G.add_nodes_from(nodes)  # Agreguem els nodes

        while degree_list:
            degree_list.sort(reverse=True)  # Ordenem en cada iteració
            d, node = degree_list.pop(0)  # Extraiem el node de major grau

            for i in range(d):  # Conectem el node amb altres 
                neighbor_degree, neighbor = degree_list[i]
                G.add_edge(node, neighbor)  
                degree_list[i] = (neighbor_degree - 1, neighbor)  # Reduïm el grau dels veïns

            degree_list = [(dg, nd) for dg, nd in degree_list if dg > 0]  # Eliminem nodes amb grau zero
        
        return G

    def createProtectedGraphDirected(self, indegreeSequence, outdegreeSequence):
        """Crear un graf dirigit a partir d'una seqüència de graus de forma manual
           Característica: Els nodes es mantenen en la seva posició original

        Args:
            indegreeSequence (np.array): Seqüència de graus d'entrada
            outdegreeSequence (np.array): Seqüència de graus de sortida

        Returns:
            nx.graph: Graf dirigit construït
        """
        in_deg = indegreeSequence.copy()
        out_deg = outdegreeSequence.copy()
        n = len(in_deg)

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        # Llista de nodes amb graus d'entrada i sortida
        degree_list = [(out_deg[i], in_deg[i], i) for i in range(n)]

        while any(out > 0 for out, _, _ in degree_list):
            # Ordem la llista
            degree_list.sort(reverse=True)
            out_u, in_u, u = degree_list.pop(0)

            # Triem nodes entrellaçar-los 
            degree_list.sort(key=lambda x: (-x[1], x[2])) 
            connections = []
            i = 0
            while len(connections) < out_u and i < len(degree_list):
                out_v, in_v, v = degree_list[i]
                if v != u and not G.has_edge(u, v) and in_v > 0:
                    connections.append((i, v))
                i += 1

            # Afegir arestes i actualitzar graus
            for idx, v in connections:
                G.add_edge(u, v)
                out_v, in_v, _ = degree_list[idx]
                degree_list[idx] = (out_v, in_v - 1, v)
            degree_list.append((0, in_u, u))  # u ya usó todos sus out-degrees

            # Eliminar nodes de la llista amb grau zero
            degree_list = [(o, i, v) for o, i, v in degree_list if o > 0 or i > 0]

        return G

    def apply_protectionUndirected(self):
        """Aplicar K-Anonimitat en el dataset, en cas de ser grafs sense direcció

        Returns:
            List, List: Llistes dels grafs originals i protegits
        """
        timestamps = len(self.grouped_df)
        originalGraphs = list()
        protectedGraphs = list()

        # Per poder aplicar la protecció, el nombre de grafs ha de ser major o igual a k
        if self.k <= timestamps:
            # Llista de grafs originals
            for _, group in self.grouped_df:
                # Iterem a la següent snapshot
                self.iterate_graph(group)
                # Calculem el seu graf complementari
                originalGraphs.append(self.graph.copy())
            
            # Llista de grafs protegits
            PMatrix = self.compute_PMatrix(self.degreeMatrix)
            anonymizedDegrees= self.anonymizeDegrees(self.degreeMatrix, PMatrix)
            realizedDegrees = self.realizeDegrees(anonymizedDegrees)
            for row in realizedDegrees:
                graphProtected = self.createProtectedGraphUndirected(row)
                protectedGraphs.append(graphProtected)
        
        return originalGraphs, protectedGraphs

    def apply_protectionDirected(self):
        """Aplicar K-Anonimitat en el dataset, en cas de ser grafs dirigits

        Returns:
            List, List: Llistes dels grafs originals i protegits
        """
        timestamps = len(self.grouped_df)
        originalGraphs = list()
        protectedGraphs = list()

        # Per poder aplicar la protecció, el nombre de grafs ha de ser major o igual a k
        if self.k <= timestamps:
            # Llista de grafs originals
            for _, group in self.grouped_df:
                # Iterem a la següent snapshot
                self.iterate_graph(group)
                # Calculem el seu graf complementari
                originalGraphs.append(self.graph.copy())
            
            # Llista de grafs protegits
            PMatrixIn = self.compute_PMatrix(self.indegreeMatrix)
            anonymizedDegreesIn= self.anonymizeDegrees(self.indegreeMatrix, PMatrixIn)
            realizedDegreesIn = self.realizeDegrees(anonymizedDegreesIn)
            for indegrees in realizedDegreesIn:
                # Fem que els outdegrees siguin una permutació dels indegrees.
                outdegrees = indegrees.copy()
                outdegrees = np.random.permutation(indegrees)
                # Creem la versió protegida del graf i la guardem en una llista
                graphProtected = self.createProtectedGraphDirected(indegrees, outdegrees)
                protectedGraphs.append(graphProtected)
        
        return originalGraphs, protectedGraphs
