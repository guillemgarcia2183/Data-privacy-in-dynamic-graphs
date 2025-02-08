### Classe per llegir el fitxer d'entrada. 
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


class Graph:
    """Mòdul creador de grafs
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('df', 'graph', 'positions')
    def __init__(self, df):
        """Inicialització de la classe

        Args:
            df (DataFrame): Dades d'un fitxer en format dataset (From, To, Timestamp)
        """
        self.df = df
        self.graph = self.create_graph()
        self.positions = nx.spring_layout(self.graph, seed=99) # Posicions fixes dels nodes del graf

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
    
    def visualize_graph(self):
        """Visualitzar cada timestamp del graf temporal
        """
        plt.figure(figsize=(5, 5)) # Tamany de la figura 
        group_timestamps = self.df.groupby("Timestamp")
        for timestamp, group in group_timestamps:
            self.graph.clear_edges() # Netejem les arestes del anterior plot
            self.graph.add_edges_from(zip(group["From"], group["To"])) # Afegim les arestes del timestamp actual
            plt.clf()  # Netejem l'anterior plot
            plt.title(f"Graph at t = {timestamp}") # Afegim el títol del gràfic
            # Dibuixem el graf en les posicions corresponents
            nx.draw(self.graph, self.positions, with_labels=True, node_color='lightblue', edge_color='red', node_size=25, font_size=2.5, arrows=True)
            plt.pause(0.1)  # Pausar per visualitzar canvis 
        plt.show() # Mostrar el graf