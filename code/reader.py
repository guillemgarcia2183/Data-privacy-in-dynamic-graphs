### Classe per llegir el fitxer d'entrada. 
import tkinter as tk
from tkinter import filedialog

class Reader:
    """Mòdul lector de dades
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('path')
    def __init__(self):
        self.path = self.import_graph()

    @staticmethod
    def open_explorer():
        """Obrir l'explorador de fitxers perquè l'usuari importi el graf 

        Returns:
            str: Path del fitxer seleccionat
        """
        # Seleccionar un fitxer i retornar-lo
        file_path = filedialog.askopenfilename(
            title="Select a Dataset",
            filetypes=[("All Files", "*.*")]
        )
        return file_path

    def import_graph(self):
        """Importació d'un graf del teu repositori local

        Returns:
            str: Path del fitxer seleccionat
        """
        # Crear una finestra oculta 
        root = tk.Tk()
        root.withdraw()
        root.title("Selecciona un graf amb format (n1,n2,timestamp)")

        # L'usuari importa el graf des de el seu repositori local
        file_path = self.open_explorer()

        # Comprovar si s'ha seleccionat un fitxer
        while (not file_path):
            print("No és vàlid el fitxer seleccionat.") 
            file_path = self.open_explorer()

        #! TODO: Llegeix el contingut del fitxer i el posem en format graf (read_graph)
        
        return file_path


    def read_graph(self):
        pass

    def check_format(self):
        """Comprova si té tres columnes i són ints (n1,n2,timestamp)
        """
        pass