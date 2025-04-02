import unittest
import os

def load_tests_from_directory(directory):
    # Cargar tots els arxius de prova que comencen amb "test_"
    test_loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                module_name = os.path.splitext(file)[0]
                if root not in os.sys.path:
                    os.sys.path.append(root)
                suite.addTests(test_loader.loadTestsFromName(module_name))
    return suite

if __name__ == '__main__':
    test_directory = os.path.dirname(os.path.abspath(__file__))
    suite = load_tests_from_directory(test_directory)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
