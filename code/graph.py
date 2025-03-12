### Classe per llegir el fitxer d'entrada. 
import networkx as nx
# import pandas as pd
import matplotlib.pyplot as plt
# import cv2
# import numpy as np
import math
# import tkinter as tk

# VARIABLES GLOBALS
# DAY = 86400 # Un dia equival a 86400 segons

# Obtenir resolució de la pantalla
# root = tk.Tk()
# WIDTH = root.winfo_screenwidth()
# HEIGHT = root.winfo_screenheight()
# root.destroy()

class GraphProtection:
    """Mòdul creador de grafs
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('path', 'weighted', 'directed', 'df',
                 'grouped_df', 'graph', 'node_positions', 'filename')
    def __init__(self, filename, input_tuple, df):
        """Inicialització de la classe

        Args:
            df (DataFrame): Dades d'un fitxer en format dataset (From, To, Timestamp)
        """
        self.path, self.weighted, self.directed = input_tuple
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
        #! TO UNITTEST IT 
        self.graph.clear_edges() # Netejem les arestes del anterior plot
        if self.weighted == "unweighted":
            self.graph.add_edges_from(zip(group["From"], group["To"])) # Afegim les arestes del timestamp o data actual
        else:
            edges = zip(self.df["From"], self.df["To"], self.df["Weight"])
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

class LEDP(GraphProtection):
    __slots__ = ('density')
    def __init__(self, filename, input_tuple, df):
        super().__init__(filename, input_tuple, df)
        self.density = self.compute_density()
        
    def compute_density(self):
        """Calcular la densitat mitjana de tots els grafs que conforma un dataset

        Returns:
            float: Densitat mitjana
        """
        density = 0
        n = 0
        for _, group in self.grouped_df:
            self.iterate_graph(group)
            density += nx.density(self.graph)
            n += 1
        return density/n
    
    def compute_probabilities(self, epsilon):
        """Calcular les probabilitats que s'usaràn per fer els noise-graphs

        Args:
            epsilon (float): Paràmetre Epsilon Local Edge Differential Privacy. Segons el seu valor, els grafs tindràn més o menys soroll.

        Returns:
            p0, p1: Probabilitats d'afegir o treure arestes pels noise-graphs
        """
        p0 = 1 - (1 / ((math.exp(epsilon) - 1 + (1 / self.density))))
        p1 = (math.exp(epsilon)) / ((math.exp(epsilon) - 1 + (1 / self.density)))
        return p0, p1

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