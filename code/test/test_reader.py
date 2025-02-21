import unittest
import sys
import os
import pandas

# Afegir el directori pare 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import reader as reader_if

# VARIABLES GLOBALS
PATH1 = parent_dir + "/data/test/CollegeMsg.txt"
PATH2 = parent_dir + "/data/test/aves-sparrow-social.edges"
PATH3 = parent_dir + "/data/test/Email-Enron.txt"

class TestReader(unittest.TestCase):
    __slots__ = ('file1', 'file2', 'file3')
    def setUp(self):
        """Crea una instància de Dataset
        """
        self.file1 = reader_if.read_file((PATH1, 'unweighted'))
        self.file2 = reader_if.read_file((PATH2, 'weighted'))

    def test_files(self):
        """1. Test de lectura de fitxers 
        """
        self.assertIsInstance(self.file1, pandas.core.frame.DataFrame)
        self.assertIsInstance(self.file2, pandas.core.frame.DataFrame)
        try:
            self.file3 = reader_if.read_file((PATH3, 'unweighted'))
        except:
            self.file3 = None
        self.assertIsNone(self.file3)         

if __name__ == '__main__':
    unittest.main()
