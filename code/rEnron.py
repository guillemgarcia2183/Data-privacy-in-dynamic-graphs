### Classe per llegir el fitxer d'entrada. 
import os
import matplotlib.pyplot as plt
from collections import Counter

#! TO DO: READ REPOSITORIES 
#!      1. LOOK A DISTRIBUTION OF THE NAMING FILES
#!      2. FILTER THE IMPORTANT ONES 
#!      3. BUILD A DATASET THAT CONTAINS (FROM, TO, TIMESTAMP, MESSAGE (?), ...)

class ReaderEnron:
    __slots__ = ('path')
    def __init__(self, path):
        """Inicialització de la classe

        Args:
            path (str): Directori on es troba tot el conjunt de carpetes de Enron
        """
        self.path = path

    def count_repositories(self):
        """Fer recompte dels repositoris que té cada membre de Enron

        Returns:
            Dict: Recompte de carpetes ordenades de forma ascendent
        """
        count = Counter()
        # Iterem els repositoris principals (membres de Enron)
        for usuario in os.listdir(self.path):
            usuario_path = os.path.join(self.path, usuario)
            # Ens assegurem que el contingut és una carpeta i l'iterem
            if os.path.isdir(usuario_path):  
                for carpeta in os.listdir(usuario_path):
                    count[carpeta.lower()] += 1  # Sumem en el recompte 
        # Filtratge de repositoris: Ens quedem amb les que es repeteixen 20 cops o més
        filtered_count =  {u:v for u,v in count.items() if v >= 20} 
        # Ordenem el recompte en ordre ascendent 
        ordered_count = dict(sorted(filtered_count.items(), key=lambda item: item[1], reverse=False)) 
        return ordered_count
    
    @staticmethod
    def plot_repositories_histogram(count):
        """Graficar l'histogrames de repositoris que es conté en Enron

        Args:
            count (Dict): Recompte de carpetes ordenades de forma ascendent
        """
        folders, amount = zip(*count.items())

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh(folders, amount, color='blue')
        ax.set_xlabel("Freqüència")
        ax.set_ylabel("Nom del repositori")
        fig.text(0.5, 0.95, "Histograma de repositoris Enron Case", ha='center', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.91, "Nombre de repositoris: 150", ha='center', fontsize=12)

        plt.show()

if __name__ == "__main__":
    maildir_path = "C:/Users/garci/Desktop/UNIVERSITAT/QUART DE CARRERA/TFG/TFG-Dynamic-Graphs/code/data/maildir"  # Ruta de la carpeta principal
    ReaderEnron = ReaderEnron(maildir_path)
    count = ReaderEnron.count_repositories()
    ReaderEnron.plot_repositories_histogram(count)

