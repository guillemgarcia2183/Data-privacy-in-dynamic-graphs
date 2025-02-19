from reader import Reader
#from graph import Graph
from datasets import DATASET1,DATASET2,DATASET3,DATASET4,DATASET5


class MainManager:
    """Classe que connecta tots els mòduls de l'aplicació
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('reader', 'graph')
    def __init__(self):
        """Gestió del procés del programa
        """
        self.introduce_program()
        selected_datasets = self.select_dataset()
        self.reader = Reader(selected_datasets)
        self.reader.retrieve_df_information()
        # self.graph = Graph(self.reader.filename, self.reader.df)
        #self.graph.visualize_per_timestamp()
        #self.graph.animate_graph()

    def introduce_program(self):
        """Dona la benvolguda al usuari i descriu el que és capaç de fer el programa
        """
        print("#################################################")
        print("PRIVACY AND COMMUNITY DETECTION FOR DYNAMIC GRAPHS")
        print("################################################# \n")
        print("El següent programa es pot utilitzar com a eina per")
        print("fer anàlisi de grafs dinàmics. Permet protegir un graf d'entrada")
        print("i fer detecció de comunitats amb diferents algorismes implementats. \n")
    
    def select_dataset(self):
        """L'usuari selecciona algun/tots els datasets que es tenen per defecte

        Returns:
            List[Tuple]: Llista amb els datasets a analitzar. Les tuples són de format (PATH, WEIGHTED, DIRECTION)
        """
        dictionary_options = {'1': (DATASET1, 'weighted', 'undirected'), 
                               '2': (DATASET2, 'unweighted', 'undirected'),
                               '3': (DATASET3, 'weighted', 'undirected'),
                               '4': (DATASET4, 'unweighted', 'directed'),
                               '5': (DATASET5, 'weighted', 'directed')}

        print("La llista de datasets que utilitzem per defecte són els següents:")
        print("(1): Aves-sparrow dataset (|V| = 52, |E| = 516, weighted, undirected)")
        print("(2): Reptilia-tortoise dataset (|V| = 45, |E| = 134, unweighted, undirected)")
        print("(3): Insecta-ant dataset (|V| = 152, |E| = 194K, weighted, undirected)")
        print("(4): CollegeMsg dataset (|V| = 1899, |E| = 59.8K, unweighted, directed)")
        print("(5): IA-Facebook dataset (|V| = 42.4K, |E| = 877K, weighted, directed) \n" )
                
        print("Explorar tots els datasets (1) // Seleccionar un dataset individualment (2): ")
        selection = input("Selecciona l'opció que vols triar (1-2): ")
        while selection not in ['1', '2']:
            selection = input("Opció incorrecte. Torna a seleccionar una de les opcions possibles (1-2): ")

        if selection == '1':
            return list(dictionary_options.values()) 
        
        option2 = input("Tria un dels datasets (1-5): ")
        while option2 not in ['1', '2', '3', '4', '5']:
            option2 = input("Tria un dels datasets (1-5): ")

        return list(dictionary_options[option2]) 


if __name__ == "__main__":
    class1 = MainManager()