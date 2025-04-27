import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Per llegir els fitxers de dades
DATASET1 = current_dir + '/data/01-easy/aves-sparrow-social.edges'
DATASET2 = current_dir + '/data/02-medium/mammalia-voles-rob-trapping.edges'
DATASET3 = current_dir + '/data/02-medium/insecta-ant-colony5.edges'
DATASET4 = current_dir + '/data/03-hard/ia-enron-employees.edges'
DATASET5 = current_dir + '/data/03-hard/CollegeMsg.txt'
#DATASET5 = current_dir + '/data/03-hard/ia-facebook-wall-wosn-dir.edges'
DATASET6 = current_dir + '/data/02-medium/LNetwork.json'

# Per llegir els fitxers de sortida quan s'apliquen els algorismes de protecció
OUTPUT_ORIGINAL = current_dir + '/output/original_graphs'
OUTPUT_ELDP = current_dir + '/output/ELDP'
OUTPUT_KDA = current_dir + '/output/KDA'
OUTPUT_KDA_RANDOM = current_dir + '/output/KDA_RANDOM'
OUTPUTS = [OUTPUT_ORIGINAL, OUTPUT_ELDP, OUTPUT_KDA, OUTPUT_KDA_RANDOM]