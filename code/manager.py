from reader import Reader
from graph import GraphProtection
import data_paths as dp

class ModuleManager:
    """Classe que connecta tots els mòduls de l'aplicació
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('datasets')
    def __init__(self):
        """Gestió del procés del programa
        """
        self.introduce_program() # Introduïm al usuari com funciona el programa
        
        self.datasets = list()
        selected_datasets = self.select_dataset() # Decidim quins són els datasets que s'utilitzaràn
        for df in selected_datasets:
            self.datasets.append(Reader(df))
        
        mode = self.select_mode() # Es selecciona el mode que volem executar
        if mode == '1':
            protection = self.select_protection()
            #! Crear segons protection K-Anonimity Class // LEDP Class
            #! Les classes de forma __init__ ja es desenvolupen la resta
        
    def introduce_program(self):
        """Dona la benvolguda al usuari i descriu el que és capaç de fer el programa
        """
        print("#################################################")
        print("PRIVACY AND COMMUNITY DETECTION FOR DYNAMIC GRAPHS")
        print("################################################# \n")
        print("El següent programa es pot utilitzar com a eina per")
        print("fer anàlisi de grafs dinàmics. Permet protegir un graf d'entrada")
        print("i fer detecció de comunitats amb diferents algorismes implementats. \n")

    
    def select_option(self, pretext, options):
        print(pretext)
        for key,value in options.items():
            print(f"({key}): {value}")
        

    def check_valid_input(self, prompt, valid_options):
        """Comprova que els inputs de l'usuari són correctes, i els torna a demanar en cas d'error

        Args:
            prompt (str): Input seleccionat
            valid_options (str): Opcions vàlides de input

        Returns:
            prompt (str): Input vàlid
        """
        while prompt not in valid_options:
            prompt = input("Opció incorrecte. Torna a seleccionar una de les opcions possibles: ")
        return prompt

#! OPTIMITZAR - FER NOMÉS UNA FUNCIÓ QUE NOMÉS HAGI D'ITERAR
    def select_dataset(self):
        """L'usuari selecciona algun/tots els datasets que es tenen per defecte

        Returns:
            List[Tuple]: Llista amb els datasets a analitzar. Les tuples són de format (PATH, WEIGHTED, DIRECTION)
        """
        dictionary_options = {'1': (dp.DATASET1, 'weighted', 'undirected'), 
                               '2': (dp.DATASET2, 'weighted', 'undirected'),
                               '3': (dp.DATASET3, 'weighted', 'undirected'),
                               '4': (dp.DATASET4, 'unweighted', 'directed'),
                               '5': (dp.DATASET5, 'weighted', 'directed')}

        print("La llista de datasets que utilitzem per defecte són els següents:")
        print("(1): Aves-sparrow dataset (|V| = 52, |E| = 516, weighted, undirected)")
        print("(2): Mammalia-voles dataset (|V| = 1480, |E| = 4569, weighted, undirected)")
        print("(3): Insecta-ant dataset (|V| = 152, |E| = 194K, weighted, undirected)")
        print("(4): CollegeMsg dataset (|V| = 1899, |E| = 59.8K, unweighted, directed)")
        print("(5): IA-Facebook dataset (|V| = 42.4K, |E| = 877K, weighted, directed) \n" )
                
        print("Explorar tots els datasets (1) // Seleccionar un dataset individualment (2): ")
        selection = input("Selecciona l'opció que vols triar (1-2): ")
        while selection not in ['1', '2']:
            selection = input("Opció incorrecte. Torna a seleccionar una de les opcions possibles (1-2): ")
        print("")
        if selection == '1':
            return list(dictionary_options.values()) 
        
        list_option2 = list()
        option2 = input("Tria un dels datasets (1-5): ")
        while option2 not in ['1', '2', '3', '4', '5']:
            option2 = input("Tria un dels datasets (1-5): ")
        list_option2.append(dictionary_options[option2])
        return list_option2

    def select_mode(self):
        """Menú d'opcions que pot fer el programa

        Returns:
            str: Opció triada: ['1', '2']
        """
        print("Menú d'opcions disponibles:")
        print("(1): Graph protection")
        print("(2): Graph prediction")
        selection = input("Selecciona un mode (1-2): ")
        while selection not in ['1', '2']:
            selection = input("Opció incorrecte. Torna a seleccionar una de les opcions possibles (1-2): ")
        print("")
        return selection

    def select_protection(self):
        """Menú d'opcions que pot fer el programa

        Returns:
            str: Opció triada: ['1', '2']
        """
        print("Modes de proteccions:")
        print("(1): K-Anonimity")
        print("(2): Local-Edge Differential Privacy")
        selection = input("Selecciona un mode (1-2): ")
        while selection not in ['1', '2']:
            selection = input("Opció incorrecte. Torna a seleccionar una de les opcions possibles (1-2): ")
        print("")
        return selection
                

if __name__ == "__main__":
    ModuleManager()