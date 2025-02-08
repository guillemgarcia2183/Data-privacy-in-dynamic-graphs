### Classe per llegir el fitxer d'entrada. 
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# VARIABLES GLOBALS
DAY = 86400 # Un dia equival a 86400 segons

class Graph:
    """Mòdul creador de grafs
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('filename', 'df', 'graph', 'positions')
    def __init__(self, filename, df):
        """Inicialització de la classe

        Args:
            df (DataFrame): Dades d'un fitxer en format dataset (From, To, Timestamp)
        """
        self.filename = filename
        self.df = df
        self.graph = self.create_graph()
        self.positions = nx.spectral_layout(self.graph) # Posicions fixes dels nodes del graf

    def create_graph(self):
        #! TO TEST IT: Len(nodes) for all datasets
        """Creació de tots els nodes del graf.  

        Returns:
            nx.graph: Graf generat sense cap relació entre nodes
        """
        nodes = set(self.df["From"]).union(set(self.df["To"])) # Fem l'unió de tots els nodes únics del dataset
        graph = nx.DiGraph() # Creem un graf dirigit amb networkx
        graph.add_nodes_from(nodes) # Afegim els nodes en el graf 
        return graph
    
    def visualize_graph(self, grouped_df):
        plt.figure(figsize=(30, 30)) # Tamany de la figura 
        for time, group in grouped_df:
            self.graph.clear_edges() # Netejem les arestes del anterior plot
            self.graph.add_edges_from(zip(group["From"], group["To"])) # Afegim les arestes del timestamp o data actual
            plt.clf()  # Netejem l'anterior plot
            plt.title(f"Plotting {self.filename} temporal graph") # Afegim el títol del gràfic
            plt.text(0.51, 0.88, f"Exchanges of messages at time: {time}", ha = "center", va="top", transform=plt.gcf().transFigure) # Afegim un subtítol 
            # Dibuixem el graf en les posicions corresponents
            nx.draw(self.graph, self.positions, with_labels=True, node_color='lightblue', edge_color='red', node_size=50, font_size=3, arrows=True, width=0.5)
            plt.pause(0.1)  # Pausar per visualitzar canvis 
        plt.show() # Mostrar el graf

    def visualize_per_timestamp(self):
        """Visualitzar cada timestamp del graf temporal
        """
        grouped_df = self.df.groupby("Timestamp")
        self.visualize_graph(grouped_df)

    def visualize_per_day(self):
        """Visualitzar el graf temporal agrupat per dies 
        """
        self.df["Date"] = pd.to_datetime(self.df["Timestamp"], unit="s").dt.date
        grouped_df = self.df.groupby("Date")
        self.visualize_graph(grouped_df)

    