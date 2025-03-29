import networkx as nx
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np

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
            df (DataFrame): Dades d'un fitxer en format dataset (From, To, Timestamp)
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
        if self.directed == "directed":
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
        if self.weighted == "unweighted":
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
        if self.directed == "directed":
            graph = nx.erdos_renyi_graph(self.nodes, p, directed=True)
        else: 
            graph = nx.erdos_renyi_graph(self.nodes, p, directed=False) 

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
    __slots__ = ('k', 'm', 'T', 'degree_matrix', 'indegree_matrix', 'outdegree_matrix')
    def __init__(self, filename, input_tuple, df, k=2):
        super().__init__(filename, input_tuple, df)
        self.k = k
        self.m = math.ceil(self.graph.number_of_nodes() / self.k)
        self.T = len(self.grouped_df)
        self.degree_matrix, self.indegree_matrix, self.outdegree_matrix = self.get_degree_matrix()

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

    def get_degree_matrix(self):
        """Obtenir les matrius de graus dels grafs. Cada fila correspon els graus d'un graf

        Returns:
            np.array(), np.array(), np.array(): Matrius degree, in_degree i out_degree
        """
        degree_matrix = None
        indegree_matrix = None
        outdegree_matrix = None

        # En cas de ser un graf dirigit, obtenir les in_degree i out_degree matrices
        if self.directed == "directed":
            for _, group in self.grouped_df:
                self.iterate_graph(group)
                indegree_array = self.get_degree("indegree")
                outdegree_array = self.get_degree("outdegree")

                if indegree_matrix is None:
                    indegree_matrix = indegree_array 
                else:
                    indegree_matrix = np.vstack((indegree_matrix, indegree_array))

                if outdegree_matrix is None:
                    outdegree_matrix = outdegree_array
                else:
                    outdegree_matrix = np.vstack((outdegree_matrix, outdegree_array))

            return None, indegree_matrix, outdegree_matrix

        # En cas de ser no dirigit, fer només una matriu de graus
        for _, group in self.grouped_df:
            self.iterate_graph(group)
            degree_array = self.get_degree("degree")
            if degree_matrix is None:
                degree_matrix = degree_array
            else:
                degree_matrix = np.vstack((degree_matrix, degree_array))

        return degree_matrix, None, None

    def compute_P_matrix(self, degree_matrix):
        """Calcular matriu de medianes P 

        Args:
            degree_matrix (np.array): Matriu de graus a particionar

        Returns:
            np.array() : Matriu de medianes 
        """
        aux_matrix = degree_matrix.copy()
        P_matrix = np.zeros((self.T, self.m))
        for i, d_seq in enumerate(aux_matrix):
            # Aleatoritzem la seqüència de graus
            np.random.shuffle(d_seq)
            # Particionem la seqüència en m particions
            particions = np.array_split(d_seq, self.m)
            # Calculem la mediana de cada partició, i l'incorporem a la matriu final
            P_matrix[i] = np.array([np.median(p) for p in particions])
        return P_matrix

    def apply_protection(self):
        pass

