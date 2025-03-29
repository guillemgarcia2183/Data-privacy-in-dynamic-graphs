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
            for index, reader in enumerate(self.datasets):
              graf = GraphProtection(reader.filename, selected_datasets[index], reader.df)
              break
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
        """Seleccionador d'opcions dels menús

        Args:
            pretext (str): Text anterior a introduïr les opcions
            options (Dict): Diccionari on les claus determinen el número d'opció, i el valor quina opció es tracta

        Returns:
            _type_: _description_
        """
        print(pretext)
        for key,value in options.items():
            print(f"({key}): {value}")
        selection = input("Selecciona l'opció que vols triar (1-" + str(len(options)) + "): ")
        print()
        return self.check_valid_input(selection, list(options.keys()))

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

    def select_dataset(self):
        """L'usuari selecciona algun/tots els datasets que es tenen per defecte

        Returns:
            List[Tuple]: Llista amb els datasets a analitzar. Les tuples són de format (PATH, WEIGHTED, DIRECTION)
        """
        dictionary_options = {'1': (dp.DATASET1, 'weighted', 'undirected', 'FILE'), 
                               '2': (dp.DATASET2, 'weighted', 'undirected', 'FILE'),
                               '3': (dp.DATASET3, 'weighted', 'undirected', 'FILE'),
                               '4': (dp.DATASET4, 'unweighted', 'directed', 'FILE'),
                               '5': (dp.DATASET5, 'weighted', 'directed', 'FILE'),
                               '6': (dp.DATASET6, 'unweighted', 'undirected', 'JSON')}

        
        print_options = {'1': "Aves-sparrow dataset (|V| = 52, |E| = 516, weighted, undirected)", 
                               '2': "Mammalia-voles dataset (|V| = 1480, |E| = 4569, weighted, undirected)",
                               '3': "Insecta-ant dataset (|V| = 152, |E| = 194K, weighted, undirected)",
                               '4': "CollegeMsg dataset (|V| = 1899, |E| = 59.8K, unweighted, directed)",
                               '5': "IA-Facebook dataset (|V| = 42.4K, |E| = 877K, weighted, directed)",
                               '6': "Lighning network dataset (JSON FORMAT)",
                               "7": "TOTS ELS DATASETS"}

        pretext = "Les opcions de datasets que es tenen són les següents:"
        selection = self.select_option(pretext, print_options)

        if selection == '7':
            return list(dictionary_options.values())
        final_list = list()
        final_list.append(dictionary_options[selection])         
        return final_list


    def select_mode(self):
        """Menú d'opcions que pot fer el programa

        Returns:
            str: Opció triada: ['1', '2']
        """
        pretext = "Menú d'opcions disponibles:"
        options = {"1": "Graph protection",
                   "2": "Graph prediction"}
        return self.select_option(pretext, options)

    def select_protection(self):
        """Menú d'opcions que pot fer com a protecció

        Returns:
            str: Opció triada: ['1', '2']
        """
        pretext = "Menú d'opcions disponibles:"
        options = {"1": "K-Anonimity",
                   "2": "Local-Edge Differential Privacy"}
        return self.select_option(pretext, options)

                
if __name__ == "__main__":
    ModuleManager()