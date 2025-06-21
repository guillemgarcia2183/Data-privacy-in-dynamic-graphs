import networkx as nx
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

class GraphProtection:  
    """Mòdul creador de grafs
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('path', 'weighted', 'directed', 'format', 'df',
                 'grouped_df', 'graph', 'node_positions', 'filename', 'grouping_option')
    def __init__(self, filename, input_tuple, df, group_option=None):
        """Inicialització de la classe

        Args:
            filename (str): Nom del arxiu
            input_tuple (tuple): Tuple amb la informació del dataset (path, weighted, directed, format)
            df (pd.DataFrame): Dataframe amb la informació del dataset
            group_option (str, optional): Opció de agrupació del dataset. Per defecte és None.
        """
        self.path, self.weighted, self.directed, self.format = input_tuple
        self.filename = filename
        self.df = df
        self.grouping_option = group_option
        self.grouped_df = self.get_grouped_df()        
        self.graph = self.create_graph()
        self.node_positions = nx.spectral_layout(self.graph) # Posicions fixes dels nodes del graf

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
        """Agrupar el dataset per timestamp o bé per dies/setmanes/mesos si aquest és gran.

        Returns:
            pd.DataFrame: Dataset agrupat
        """
        if self.grouping_option is None:
            grouped_df = self.df.groupby("Timestamp")
            return grouped_df
        grouped_df = self.groupby_option(self.grouping_option)
        return grouped_df
    
    def groupby_option(self, option):
        """Depenent de l'opció, agrupar un dataset per hores, dies o setmanes 

        Args:
            option (str): Opció escollida per l'usuari

        Returns:
            pd.DataFrame: Dataset agrupat per hores/dies/setmanes 
        """
        
        self.df["Date"] = pd.to_datetime(self.df["Timestamp"], unit="s")
        if option == "1":
            self.filename = "HOUR_" + self.filename
            return self.groupby_hour()
        elif option == "2":
            self.filename = "DAY_" + self.filename
            return self.groupby_day()
        elif option == "3":
            self.filename = "WEEK_" + self.filename
            return self.groupby_week()
        elif option == "4":
            self.filename = "MONTH_" + self.filename
            return self.groupby_month() 
        elif option == "5":
            self.filename = "YEAR_" + self.filename
            return self.groupby_year() 

    def groupby_hour(self):
        """Agrupar el dataset per hores

        Returns:
            pd.DataFrame: Dataset agrupat per hores
        """
        df_by_hour = self.df.groupby(self.df["Date"].dt.floor('H'))
        return df_by_hour

    def groupby_day(self):
        """Agrupar el dataset per dies

        Returns:
            pd.DataFrame: Dataset agrupat per dies
        """
        df_by_day  = self.df.groupby(self.df["Date"].dt.date)
        return df_by_day 
    
    def groupby_week(self):
        """Agrupar el dataset per setmanes

        Returns:
            pd.DataFrame: Dataset agrupat per setmanes
        """
        df_by_week = self.df.groupby(self.df["Date"].dt.to_period('W').apply(lambda r: r.start_time))
        return df_by_week
    
    def groupby_month(self):
        """Agrupar el dataset per mesos

        Returns:
            pd.DataFrame: Dataset agrupat per mesos
        """
        df_by_month = self.df.groupby(self.df["Date"].dt.to_period('M').apply(lambda r: r.start_time))
        return df_by_month

    def groupby_year(self):
        """Agrupar el dataset per anys

        Returns:
            pd.DataFrame: Dataset agrupat per anys
        """
        df_by_year = self.df.groupby(self.df["Date"].dt.to_period('Y').apply(lambda r: r.start_time))
        return df_by_year
    
    def iterate_graph(self, group):
        """Canviar de snapshot del graf temporal, treient les arestes del timestamp actual i afegint les del següent.

        Args:
            group (pd.DataFrame): Dataset que conté les arestes del timestamp actual
        """
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
        
    def save_graphs(self, original_graphs, protected_graphs, algorithm, experiment, parameter):
        # Canvi de directori al repositori de l'aplicació
        directoryFiles = {"aves-sparrow-social.edges":"AVES-SPARROW",
                          "insecta-ant-colony5.edges": "INSECTA-ANT",
                          "mammalia-voles-rob-trapping.edges": "MAMMALIA-VOLES"}
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        actualRepository = os.getcwd()

        og = actualRepository + "/output/original_graphs/" + str(self.filename) + ".pkl"
        newRepository = newRepository = os.path.join(actualRepository, "output", str(algorithm), str(experiment), directoryFiles[str(self.filename)])
        pg =  os.path.join(newRepository, f"{self.filename}_{parameter}.pkl")

        if not os.path.exists(newRepository):
            os.makedirs(newRepository)
        
        with open(og, "wb") as f:
            pickle.dump(original_graphs, f)

        with open(pg, "wb") as f:
            pickle.dump(protected_graphs, f)
            
class ELDP(GraphProtection):
    __slots__ = ('nodes', 'epsilon')
    def __init__(self, filename, input_tuple, df, group_option=None, epsilon=0.5):
        """Inicialització del mòdul ELDP

        Args:
            filename (str): Nom del arxiu
            input_tuple (tuple): Tuple amb la informació del dataset (path, weighted, directed, format)
            df (pd.DataFrame): Dataframe amb la informació del dataset
            group_option (str, optional): Opció de agrupació del dataset. Per defecte és None.
            epsilon (float): Paràmetre epsilon utilitzat per l'algoritme ELDP. Per defecte és 0.5.
        """
        super().__init__(filename, input_tuple, df, group_option)
        self.epsilon = epsilon
        self.nodes = self.graph.number_of_nodes()

    def compute_density(self):
        """Calcular la densitat del graf actual

        Returns:
            float: Densitat mitjana
        """
        density = nx.density(self.graph)
        return density
    
    def compute_probabilities(self, density):
        """Calcular les probabilitats que s'usaràn per fer els noise-graphs

        Args:
            density (float): Densitat del graf actual

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
            nx.graph: Graf resultant de la suma XOR
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

        for _, group in tqdm(self.grouped_df, desc="Aplicant ELDP " + str(self.filename) + ": ε = " + str(self.epsilon)):
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
    def __init__(self, filename, input_tuple, df, group_option=None, k=2):
        """_summary_

        Args:
            filename (str): Nom del arxiu
            input_tuple (tuple): Tuple amb la informació del dataset (path, weighted, directed, format)
            df (pd.DataFrame): Dataframe amb la informació del dataset
            group_option (str, optional): Opció de agrupació del dataset. Per defecte és None.
            k (int): Paràmetre k que s'utilitza en k-anonymity. Per defecte és 2
        """
        super().__init__(filename, input_tuple, df, group_option)
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

    def compute_PMatrix(self, degreeMatrix, randomize=True):
        """Calcular matriu de medianes P 

        Args:
            degreeMatrix (np.array): Matriu de graus a particionar
            randomize (bool): Si és True, aleatoritzar les particions de la matriu de medianes PMatrix

        Returns:
            np.array() : Matriu de medianes 
        """
        aux_matrix = degreeMatrix.copy()
        PMatrix = np.zeros((self.T, self.m), dtype=int)
        for i, d_seq in enumerate(aux_matrix):
            # Aleatoritzem la seqüència de graus, en cas de voler fer el procediment d'aquesta forma
            if randomize:
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
        if dictNorealizable:
            # Iterem totes les seqüències no realizables
            for k1, v1 in dictNorealizable.items():
                distances = {}
                # Iterem totes les seqüències realizables, i veiem quina és la que menys distància té
                for k2,v2 in dictRealizable.items():
                    distances[k2] = self.compute_l1Distance(degreeMatrix[v1[0]], degreeMatrix[v2[0]]) 
                
                # Trobar la key de la seqüència realizable amb la mínima distància
                try:
                    min_key = min(distances, key=distances.get)
                    dictRealizable[min_key] += v1
                except:
                    pass            

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

    def apply_protectionUndirected(self, randomize=True):
        """Aplicar K-Anonimitat en el dataset, en cas de ser grafs sense direcció

        Args:
            randomize (bool): Si True, aleatoritzar les particions de la matriu de medianes PMatrix

        Returns:
            List, List: Llistes dels grafs originals i protegits
        """
        timestamps = len(self.grouped_df)
        originalGraphs = list()
        protectedGraphs = list()

        # Per poder aplicar la protecció, el nombre de grafs ha de ser major o igual a k
        if self.k <= timestamps:
            # Llista de grafs originals
            for _, group in tqdm(self.grouped_df, desc="Creant grafs originals " + str(self.filename)):
                # Iterem a la següent snapshot
                self.iterate_graph(group)
                # Calculem el seu graf complementari
                originalGraphs.append(self.graph.copy())
            
            # Llista de grafs protegits
            PMatrix = self.compute_PMatrix(self.degreeMatrix, randomize)
            anonymizedDegrees= self.anonymizeDegrees(self.degreeMatrix, PMatrix)
            realizedDegrees = self.realizeDegrees(anonymizedDegrees)
            for row in tqdm(realizedDegrees, desc="Creant grafs protegits " + str(self.filename)):
                graphProtected = self.createProtectedGraphUndirected(row)
                protectedGraphs.append(graphProtected)
        
        return originalGraphs, protectedGraphs

    def apply_protectionDirected(self, randomize=True):
        """Aplicar K-Anonimitat en el dataset, en cas de ser grafs dirigits

        Args:
            randomize (bool): Si True, aleatoritzar les particions de la matriu de medianes PMatrix

        Returns:
            List, List: Llistes dels grafs originals i protegits
        """
        timestamps = len(self.grouped_df)
        originalGraphs = list()
        protectedGraphs = list()

        # Per poder aplicar la protecció, el nombre de grafs ha de ser major o igual a k
        if self.k <= timestamps:
            # Llista de grafs originals
            for _, group in tqdm(self.grouped_df, desc="Creant grafs originals " + str(self.filename)):
                # Iterem a la següent snapshot
                self.iterate_graph(group)
                # Calculem el seu graf complementari
                originalGraphs.append(self.graph.copy())
            
            print()
            # Llista de grafs protegits
            PMatrixIn = self.compute_PMatrix(self.indegreeMatrix, randomize)
            anonymizedDegreesIn= self.anonymizeDegrees(self.indegreeMatrix, PMatrixIn)
            realizedDegreesIn = self.realizeDegrees(anonymizedDegreesIn)
            for indegrees in tqdm(realizedDegreesIn, desc="Creant grafs protegits " + str(self.filename)):
                # Fem que els outdegrees siguin una permutació dels indegrees.
                outdegrees = indegrees.copy()
                outdegrees = np.random.permutation(indegrees)
                # Creem la versió protegida del graf i la guardem en una llista
                graphProtected = self.createProtectedGraphDirected(indegrees, outdegrees)
                protectedGraphs.append(graphProtected)
        else:
            print(f"El valor de k ({self.k}) és superior al nombre de grafs ({timestamps}). NO es pot aplicar la protecció")
        
        return originalGraphs, protectedGraphs

    def apply_protection(self, randomize=True):
        """Aplicar K-Anonimitat en el dataset, en funció de si és dirigit o no

        Args:
            randomize (bool): Si True, aleatoritzar les particions de la matriu de medianes PMatrix

        Returns:
            List, List: Llistes dels grafs originals i protegits
        """
        if self.directed:
            return self.apply_protectionDirected(randomize)
        else:
            return self.apply_protectionUndirected(randomize)