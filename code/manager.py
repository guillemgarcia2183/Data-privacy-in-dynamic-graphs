from reader import Reader
from graph import GraphProtection, KDA, ELDP
import data_paths as dp
from tqdm import tqdm

class ModuleManager:
    """Classe que connecta tots els mòduls de l'aplicació
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('datasets', 'dictionary_options', 'options', 'grouping_option')
    def __init__(self):
        """Gestió del procés del programa
        """
        self.dictionary_options = {'1': (dp.DATASET1, True, False, 'FILE'), 
                        '2': (dp.DATASET2, True, False, 'FILE'),
                        '3': (dp.DATASET3, True, False, 'FILE'),
                        '4': (dp.DATASET4, True, True, 'FILE'),
                        '5': (dp.DATASET5, False, True, 'FILE')}
        
        self.options = None
        self.grouping_option = None

        self.introduce_program() # Introduïm al usuari com funciona el programa
        
        mode = self.select_mode() # Es selecciona el mode que volem executar
        
        self.datasets = list()
        selected_datasets = self.select_dataset() # Decidim quins són els datasets que s'utilitzaràn
        for df in selected_datasets:
            self.datasets.append(Reader(df))

        self.execute_mode(mode) # Executem el mode seleccionat        
        
    def introduce_program(self):
        """Dona la benvolguda al usuari i descriu el que és capaç de fer el programa
        """
        print("#################################################")
        print("PRIVACY AND COMMUNITY DETECTION FOR DYNAMIC GRAPHS")
        print("################################################# \n")
        print("El següent programa es pot utilitzar com a eina per")
        print("fer anàlisi de grafs dinàmics. Permet protegir un dataset d'entrada,")
        print("també calcular mètriques de grafs, i per últim fer predicció amb GLMs. \n")

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
        print_options = {'1': "Aves-sparrow (|V| = 52, |E| = 516, weighted, undirected)", 
                               '2': "Mammalia-voles (|V| = 1480, |E| = 4569, weighted, undirected)",
                               '3': "Insecta-ant (|V| = 152, |E| = 194K, weighted, undirected)",
                               '4': "Enron-Employees (|V| = 150, |E| = 50.5K, weighted, directed)",
                               '5': "CollegeMsg (|V| = 1899, |E| = 59.8K, unweighted, directed)",
                               "6": "TOTS ELS DATASETS"}

        pretext = "Les opcions de datasets que es tenen són les següents:"
        selection = self.select_option(pretext, print_options)
        
        if selection in ["4", "5", "6"]:
            pretext = "Selecciona el tipus de agrupament temporal que vols aplicar als grafs grans"
            options = {"1": "Agrupació per hores",
                       "2": "Agrupació per dies",
                       "3": "Agrupació per setmanes",
                       "4": "Agrupació per mesos",
                       "5": "Agrupació per anys"}
            self.grouping_option = self.select_option(pretext, options)

        if selection == '6':
            self.options = list(self.dictionary_options.keys())
            return list(self.dictionary_options.values())
            
        final_list = list()
        final_list.append(self.dictionary_options[selection])
        self.options = list(selection)         
        return final_list


    def select_mode(self):
        """Menú d'opcions que pot fer el programa

        Returns:
            str: Opció triada: ['1', '2']
        """
        pretext = "Menú d'opcions disponibles:"
        
        options = {"1": "Graph protection",
                   "2":"Metric computation",
                   "3": "Graph prediction"}
        
        return self.select_option(pretext, options)

    def select_protection(self):
        """Menú d'opcions que pot fer com a protecció

        Returns:
            str: Opció triada: ['1', '2']
        """
        pretext = "Selecciona un mètode de protecció:"
        options = {"1": "K-Anonimity",
                   "2": "Edge-Local Differential Privacy"}
        return self.select_option(pretext, options)

    def execute_KAnonimity(self):
        """Realitzar la protecció K-Anonimity sobre els datasets seleccionats
        """
        k = 1
        random = None
        type_KDA = None

        # Preguntar per si es vol aletorietat en les particions
        while random is None:
            randomize = input("Vols que s'apliqui aletorietat al fer K-Anonimity? (S/N): ")
            if randomize.upper() == 'S':
                random = True
                type_KDA = "KDA_RANDOM"
            elif randomize.upper() == 'N':
                random = False
                type_KDA = "KDA"

        # Introduir una k vàlida
        while k < 2:
            try:  
                k = int(input("Introdueix el valor de k: "))
            except:
                print("Valor incorrecte. Torna a introduir un valor enter. \n")
        
        print()

        for reader, idx in zip(self.datasets, self.options):
            # print(f"Options: {self.dictionary_options[idx]}")

            graph = KDA(reader.filename, self.dictionary_options[idx], reader.df, self.grouping_option, k)
            orignalGraphs, protectedGraphs = graph.apply_protection(randomize=random)
            if len(orignalGraphs) > 0 and len(protectedGraphs) > 0:
                graph.save_graphs(orignalGraphs, protectedGraphs, type_KDA, k)
        
        print("Protecció KDA aplicada amb èxit. Ves al directori code/output per veure els grafs guardats! \n")

    
    def execute_ELDP(self):
        """Realitzar la protecció ELDP sobre els datasets seleccionats
        """
        epsilon = 0
        # Introduir una epsilon vàlida
        while epsilon <= 0:
            try:  
                epsilon = int(input("Introdueix el valor de epsilon: "))
            except:
                print("Valor incorrecte. Torna a introduir un valor. \n")

        print()

        for reader, idx in zip(self.datasets, self.options):
            # print(f"Options: {self.dictionary_options[idx]}")

            graph = ELDP(reader.filename, self.dictionary_options[idx], reader.df, self.grouping_option, epsilon)
            orignalGraphs, protectedGraphs = graph.apply_protection()
            if len(orignalGraphs) > 0 and len(protectedGraphs) > 0:
                graph.save_graphs(orignalGraphs, protectedGraphs, "ELDP", epsilon)

        print("Protecció ELDP aplicada amb èxit. Ves al directori code/output per veure els grafs guardats! \n")

    def execute_mode(self, mode):
        """Executa el mode seleccionat per l'usuari

        Args:
            mode (str): Mode seleccionat: ['1', '2', '3']
        """
        if mode == '1':
            protectionMethod = self.select_protection()
            if protectionMethod == '1':
                self.execute_KAnonimity()
            else:
                self.execute_ELDP()

        #! mode == '2' i mode == '3' no estan implementats, fer-ho quan toqui

                
if __name__ == "__main__":
    ModuleManager()