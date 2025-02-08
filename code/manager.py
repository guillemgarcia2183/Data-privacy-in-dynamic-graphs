from reader import Reader
from graph import Graph

class MainManager:
    """Classe que connecta tots els mòduls de l'aplicació
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('reader', 'graph')
    def __init__(self):
        """Inicialització dels paràmetres de la classe
        """
        self.reader = Reader()
        self.reader.retrieve_df_information()
        self.graph = Graph(self.reader.df)
        self.graph.visualize_graph()