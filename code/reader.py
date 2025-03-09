### Classe per llegir el fitxer d'entrada. 
import pandas as pd

def read_file(tp):
    """Llegir un fitxer d'entrada i passar a un DataFrame de pandas

    Args:
        tp (Tuple): Tupla en format (PATH, WEIGHTED, DIRECTION). Informació rellevant sobre l'arxiu que es vol llegir 

    Raises:
        Exception: En cas de no tenir el format específic (node1, node2, (optional: weight), timestamp) es retornarà un error 

    Returns:
        DataFrame: Taula amb el contingut del fitxer
    """
    try:
        data = list()
        # En cas que el graf que estem tractant no té pesos:
        if tp[1] == 'unweighted':
            # Llegir el fitxer i carregar la informació en una taula pandas
            with open(tp[0], 'r') as file:
                for line in file:
                    from_node, to_node, timestamp = map(int, line.split())  
                    data.append({'From': from_node, 'To': to_node, 'Timestamp': timestamp})
            df = pd.DataFrame(data)
            return df
        # En cas que el graf tingui pesos
        with open(tp[0], 'r') as file:
            # Llegir el fitxer i carregar la informació en una taula pandas
            for line in file:
                from_node, to_node, weight, timestamp = map(float, line.split())  
                #print(from_node, to_node, weight, timestamp)
                data.append({'From': int(from_node), 'To': int(to_node), 'Weight': weight, 'Timestamp': int(timestamp)})
        df = pd.DataFrame(data)
        return df
    # En cas de no seguir un dels formats establerts, parar execució del programa
    except:
        raise Exception("Format no vàlid!")
    
class Reader:
    """Mòdul lector de dades
    """
    # Definició de slots per evitar la creació de noves instàncies de la classe i aprofitar memòria
    __slots__ = ('filename', 'df')
    def __init__(self, file):
        """Inicialització de la classe
        """
        self.filename = str()
        self.df = self.create_dataset(file)

    def extract_filename(self, file_path):
        """Extraure el nom del fitxer de la ruta del fitxer

        Args:
            file_path (str): Path del arxiu

        Returns:
            str: Nom del arxiu
        """
        splitted_list = file_path.split("/")
        return splitted_list[-1]

    def create_dataset(self, file):
        """Conversió dels fitxers passats com a entrada a datasets de pandas
        
        Args:
            file (List): Llista d'arxius seleccionats a passar-los a dataset 

        Returns:
            List[DataFrame]: Taules dels continguts de cada arxiu. 
        """
        self.filename = self.extract_filename(file[0])
        df = read_file(file)
        df = df.sort_values(by="Timestamp")
        return df

    def retrieve_df_information(self):
        """Mostrar informació del fitxer llegit.
        """
        ngraphs = self.df['Timestamp'].nunique()
        total_nodes = pd.concat([self.df['From'], self.df['To']]).nunique()
        total_edges = len(self.df)
        print(f"\n ############# INFORMACIÓ DEL FITXER: {self.filename} #############")
        print(f"Nombre de nodes totals |V| = {total_nodes}")
        print(f"Nombre d'arestes totals |E| = {total_edges}")
        print(f"Nombre de snapshots |t| = {ngraphs}")
            

    