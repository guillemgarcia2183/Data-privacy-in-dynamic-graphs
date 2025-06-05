import os
import pickle
import data_paths as dp  
from tqdm import tqdm

def convertOutputsTxt():
    """Converteix els grafs originals i protegits en format .txt
    """
    folders = dp.OUTPUTS
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for root_folder in folders:
        for dirpath, dirnames, filenames in tqdm(os.walk(root_folder), desc="Conversió de grafs en .txt", unit="repositori"):
            for file in filenames:
                if not file.endswith('.pkl') or file.startswith('HOUR_CollegeMsg'):
                    continue  

                input_path = os.path.join(dirpath, file)

                with open(input_path, 'rb') as f:
                    graphList = pickle.load(f)

                rel_path = os.path.relpath(dirpath, root_folder)

                base_folder = os.path.basename(root_folder)
                out_dir = os.path.join(current_dir, "data", "output", base_folder, rel_path)
                os.makedirs(out_dir, exist_ok=True)

                output_file = os.path.join(out_dir, file[:-4] + ".txt")

                with open(output_file, mode='w', encoding='utf-8') as o:
                    for timestamp, graph in enumerate(graphList):
                        for u, v in graph.edges():
                            o.write(f"{u} {v} {timestamp}\n")

if __name__ == "__main__":
    convertOutputsTxt()
