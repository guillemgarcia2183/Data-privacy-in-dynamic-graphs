### Execució el programa 
import os
from manager import ModuleManager

# Canvi de directori al repositori de l'aplicació
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    ModuleManager()
