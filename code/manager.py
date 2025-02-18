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
        print("El següent programa es pot utilitzar com a eina per")
        print("fer anàlisi de grafs dinàmics. Permet protegir un graf d'entrada")
        print("i fer detecció de comunitats amb diferents algorismes implementats. \n")

        #! TODO: Preguntar si es vol executar tots a la vegada per fer comparatives

        print("Seleccioni un dels grafs de la següent llista utilitzats com exemples:")
        print("(1): Aves-sparrow dataset (|V| = 52, |E| = 516, weighted, undirected)")
        print("(2): Reptilia-tortoise dataset (|V| = 45, |E| = 134, unweighted, undirected)")
        print("(3): Insecta-ant dataset (|V| = 152, |E| = 194K, weighted, undirected)")
        print("(4): CollegeMsg dataset (|V| = 1899, |E| = 59.8K, unweighted, directed)")
        print("(5): IA-Facebook dataset (|V| = 42.4K, |E| = 877K, weighted, directed)")
        print("################################################# \n")

        #! TODO: FER LA SELECCIÓ, ADAPTAR EL READER A LA SELECCIÓ!
        