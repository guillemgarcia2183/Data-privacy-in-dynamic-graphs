### Classe per llegir el fitxer d'entrada. 
import pandas as pd
import easygui

#! TODO: Alguns grafs són weighted i altres de unweighted. Alguns grafs són dirigits, i d'altres que són undirected.
#! Eliminar obrir directori d'arxius, millor fer un menú si les opcions són reduïdes. En cas de voler afegir-ne un de nou fora de la llista, implementar-ho! 

def read_file(file_path):
    """Llegir un fitxer d'entrada i passar a un DataFrame de pandas

    Args:
        file_path (str): Directori on es troba el fitxer

    Raises:
        Exception: En cas de no tenir el format específic (node1, node2, timestamp) es retornarà un error 

    Returns:
        DataFrame: Taula amb el contingut del fitxer
    """
    try:
        data = list()
        # Llegir el fitxer i carregar la informació en una taula pandas
        with open(file_path, 'r') as file:
            for line in file:
                from_node, to_node, timestamp = map(int, line.split())  
                data.append({'From': from_node, 'To': to_node, 'Timestamp': timestamp})
                #print(f"From {from_node} to {to_node} in timestamp: {timestamp}")
        df = pd.DataFrame(data)
        return df
    except:
        raise Exception("Format no vàlid!")
    
class Reader:
    """Mòdul lector de dades
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('filename', 'df')
    def __init__(self):
        """Inicialització de la classe
        """
        self.df = self.create_dataset()
        self.filename = ''

    # @staticmethod
    # def import_file():
    #     """Obrir l'explorador de fitxers perquè l'usuari importi el graf 

    #     Returns:
    #         str: Path del fitxer seleccionat
    #     """
    #     # Seleccionar un fitxer i retornar-lo
    #     file_path = filedialog.askopenfilename(
    #         title="Select a Dataset",
    #         filetypes=[("All Files", "*.*")]
    #     )
    #     return file_path
    
    def extract_filename(self, file_path):
        """Extraure el nom del fitxer de la ruta del fitxer
        """
        print(file_path)
        splitted_list = file_path.split("/")
        return splitted_list[-1]

    def create_dataset(self):
        """Importación de un grafo desde el repositorio local.

        Returns:
            DataFrame: Tabla con el contenido del archivo.
        """
        file_path = easygui.fileopenbox(title="Selecciona un graf dinàmic")
        
        while not file_path:
            print("No és vàlid el fitxer seleccionat.") 
            file_path = easygui.fileopenbox(title="Selecciona un graf dinàmic")

        self.filename = self.extract_filename(file_path)
        
        df = read_file(file_path)
        df = df.sort_values(by="Timestamp")  
        return df


    def retrieve_df_information(self):
        """Mostrar informació del fitxer llegit.
        """
        ngraphs = self.df['Timestamp'].nunique()
        total_nodes = pd.concat([self.df['From'], self.df['To']]).nunique()
        total_edges = len(self.df)
        print("############# INFORMACIÓ DEL FITXER #############")
        print(f"Nombre de grafs (timestamps únics): {ngraphs}")
        print(f"Nombre de nodes totals: {total_nodes}")
        print(f"Nombre d'arestes totals: {total_edges}")
        print("#################################################")


