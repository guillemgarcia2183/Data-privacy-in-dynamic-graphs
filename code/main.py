### Execució el programa 
import os
from reader import Reader

# Canvi de directori al repositori de l'aplicació
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    reader = Reader()
    # Si són molts mòduls intentaré fer un mòdul "Engine"
    # El primer pas serà llegir el fitxer amb el mòdul Reader
    # Visualització del graf (si és necessari)
    # Procés d'algorismes per privatitzar dades 

    # Testos: 
    #   Format del fitxer entrada (n1,n2,timestamp)
    #   Algorismes aplicats en els grafs.