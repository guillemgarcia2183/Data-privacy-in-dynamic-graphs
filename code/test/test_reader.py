import unittest
import sys
import os
import pandas

# Afegir el directori pare 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import rif as reader_if

# VARIABLES GLOBALS
FILE1_PATH1 = "C:/Users/garci/Desktop/UNIVERSITAT/QUART DE CARRERA/TFG/TFG-Dynamic-Graphs/code/data/CollegeMsg.txt"
FILE2_PATH1 = "C:/Users/garci/Desktop/UNIVERSITAT/QUART DE CARRERA/TFG/TFG-Dynamic-Graphs/code/data/Email-Enron.txt"
FILE1_PATH2 = "C:/Users/garci/Desktop/TFG/TFG-Dynamic-Graphs/code/data/CollegeMsg.txt"
FILE2_PATH2 = "C:/Users/garci/Desktop/TFG/TFG-Dynamic-Graphs/code/data/Email-Enron.txt"

class TestReader(unittest.TestCase):
    __slots__ = ('file1', 'file2')
    def setUp(self):
        """Crea una instància de Dataset
        """
        self.file1 = reader_if.read_file(FILE1_PATH2)

    def test_files(self):
        """1. Test de lectura de fitxers 
        """
        self.assertIsInstance(self.file1, pandas.core.frame.DataFrame)
        try:
            self.file2 = reader_if.read_file(FILE2_PATH2)
        except:
            self.file2 = None
        self.assertIsNone(self.file2)         

if __name__ == '__main__':
    unittest.main()
