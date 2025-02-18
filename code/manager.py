from rif import Reader
from graph import Graph

class MainManager:
    """Classe que connecta tots els mòduls de l'aplicació
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('reader', 'graph')
    def __init__(self):
        """Gestió del procés del programa
        """
        self.introduce_program()
        #selection = self.choose_difficulty()
        self.reader = Reader()
        self.reader.retrieve_df_information()
        self.graph = Graph(self.reader.filename, self.reader.df)
        #self.graph.visualize_per_timestamp()
        #self.graph.animate_graph()

    def introduce_program(self):
        print("#################################################")
        print("PRIVACY AND COMMUNITY DETECTION FOR DYNAMIC GRAPHS")
        print("################################################# \n")
        print("This program analyzes dynamic graphs to provide insights")
        print("on privacy risks and community structures. It helps detect")
        print("sensitive information exposure and identifies clusters of nodes")
        print("that evolve over time. \n")
        print("Features:")
        print("- Privacy risk assessment")
        print("- Community detection algorithms")
        print("- Temporal graph analysis")
        print("- Visualization of dynamic structures \n")
        print("################################################# \n")
