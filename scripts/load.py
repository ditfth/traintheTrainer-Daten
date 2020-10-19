import os
import glob
cwd = os.getcwd()
from zipfile import ZipFile
import requests
import io
import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import pandas as pd
import random
random.seed(42)
#print('test')
import pandas as pd
DOWNLOAD_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00270/"
ZIPFILE = "/driftdataset.zip"
download_url=DOWNLOAD_URL
zipfile=ZIPFILE
cwd = os.getcwd()

dat_dir = cwd + "/input/dataset_dat/"
csv_dir = cwd + "/input/dataset_csv/"

def class_label(gas_label, conc_label):
    if conc_label >= 150:
        label = str((gas_label - 1) * 3 + 3)
    elif conc_label <= 75:
        label = str((gas_label - 1) * 3 + 1)
    elif conc_label > 75 and conc_label < 150:
        label = str((gas_label - 1) * 3 + 2)
    return label

def convert_dat_to_csv(number_of_batches=10):
    n_batches = number_of_batches
    print("Convert dat files to csv")


    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    full_path_csv_merged = csv_dir + "all_batches.csv"
    
    m = open(full_path_csv_merged, 'w+')
    all_lines = 0
    
    for i in range(n_batches):
        full_path_dat = dat_dir + "batch{}.dat".format(str(i+1))
        full_path_csv = csv_dir + "batch{}.csv".format(str(i+1))

        f = open(full_path_csv, 'w+')
        with open(full_path_dat) as fp:
            lines = 0
            for cnt, line in enumerate(fp):
                line_split = line.split(';', 1)
                line_split2 = line_split[1].split(' ', 1)
                line = class_label(int(line_split[0]),int(line_split2[0].split('.', 1)[0])) + "  " + line_split2[1]
                f.write(line), m.write(line)
                lines += 1
                all_lines += 1
            print("{} lines have been written for batch {}".format(lines, i+1))
        f.close()
    m.close()
    print("{} lines have been written in total".format(all_lines))
    

    
def import_to_np(path, file="all_batches.csv"):
    print("call load_svmlight_file for {}".format(file))
    data = load_svmlight_file(path + file)
    return data[0], data[1]    


def load(limit=None):
    
    DOWNLOAD_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00270/"
    ZIPFILE = "/driftdataset.zip"
    download_url=DOWNLOAD_URL
    zipfile=ZIPFILE
    cwd = os.getcwd()

    dat_dir = cwd + "/input/dataset_dat/"
    csv_dir = cwd + "/input/dataset_csv/"
    
    
    print("Download dataset from URL: {}".format(download_url))
    r = requests.get(download_url+zipfile, stream=True)
    file = ZipFile(io.BytesIO(r.content))
    print("Extract files to : {}{}".format(dat_dir, zipfile))
    file.extractall(dat_dir)
    convert_dat_to_csv()
    header = ['target', 'DR_1', '|DR|_1', 'EMAi0.001_1', 'EMAi0.01_1', 'EMAi0.1_1', 'EMAd0.001_1', 'EMAd0.01_1', 'EMAd0.1_1', 'DR_2', '|DR|_2', 'EMAi0.001_2', 'EMAi0.01_2', 'EMAi0.1_2', 'EMAd0.001_2', 'EMAd0.01_2', 'EMAd0.1_2', 'DR_3', '|DR|_3', 'EMAi0.001_3', 'EMAi0.01_3', 'EMAi0.1_3', 'EMAd0.001_3', 'EMAd0.01_3', 'EMAd0.1_3', 'DR_4', '|DR|_4', 'EMAi0.001_4', 'EMAi0.01_4', 'EMAi0.1_4', 'EMAd0.001_4', 'EMAd0.01_4', 'EMAd0.1_4', 'DR_5', '|DR|_5', 'EMAi0.001_5', 'EMAi0.01_5', 'EMAi0.1_5', 'EMAd0.001_5', 'EMAd0.01_5', 'EMAd0.1_5', 'DR_6', '|DR|_6', 'EMAi0.001_6', 'EMAi0.01_6', 'EMAi0.1_6', 'EMAd0.001_6', 'EMAd0.01_6', 'EMAd0.1_6', 'DR_7', '|DR|_7', 'EMAi0.001_7', 'EMAi0.01_7', 'EMAi0.1_7', 'EMAd0.001_7', 'EMAd0.01_7', 'EMAd0.1_7', 'DR_8', '|DR|_8', 'EMAi0.001_8', 'EMAi0.01_8', 'EMAi0.1_8', 'EMAd0.001_8', 'EMAd0.01_8', 'EMAd0.1_8', 'DR_9', '|DR|_9', 'EMAi0.001_9', 'EMAi0.01_9', 'EMAi0.1_9', 'EMAd0.001_9', 'EMAd0.01_9', 'EMAd0.1_9', 'DR_10', '|DR|_1.1', 'EMAi0.001_10', 'EMAi0.01_10', 'EMAi0.1_10', 'EMAd0.001_10', 'EMAd0.01_10', 'EMAd0.1_10', 'DR_11', '|DR|_11', 'EMAi0.001_11', 'EMAi0.01_11', 'EMAi0.1_11', 'EMAd0.001_11', 'EMAd0.01_11', 'EMAd0.1_11', 'DR_12', '|DR|_12', 'EMAi0.001_12', 'EMAi0.01_12', 'EMAi0.1_12', 'EMAd0.001_12', 'EMAd0.01_12', 'EMAd0.1_12', 'DR_13', '|DR|_13', 'EMAi0.001_13', 'EMAi0.01_13', 'EMAi0.1_13', 'EMAd0.001_13', 'EMAd0.01_13', 'EMAd0.1_13', 'DR_14', '|DR|_14', 'EMAi0.001_14', 'EMAi0.01_14', 'EMAi0.1_14', 'EMAd0.001_14', 'EMAd0.01_14', 'EMAd0.1_14', 'DR_15', '|DR|_15', 'EMAi0.001_15', 'EMAi0.01_15', 'EMAi0.1_15', 'EMAd0.001_15', 'EMAd0.01_15', 'EMAd0.1_15', 'DR_16', '|DR|_16', 'EMAi0.001_16', 'EMAi0.01_16', 'EMAi0.1_16', 'EMAd0.001_16', 'EMAd0.01_16', 'EMAd0.1_16']
    gas = ['Ethanol low', 'Ethanol medium', 'Ethanol high',
           'Ethen low', 'Ethen medium', 'Ethen high',
           'Ammoniak low', 'Ammoniak medium', 'Ammoniak high',
           'Acetaldehyd low', 'Acetaldehyd medium', 'Acetaldehyd high',
           'Aceton low', 'Aceton medium', 'Aceton high',
           'Toluol low', 'Toluol medium', 'Toluol high']
    df_original_dir = cwd + "/input/original_df/"

    if not os.path.exists(df_original_dir):
        os.makedirs(df_original_dir)
        
    np.set_printoptions(suppress=True)

    full_array = []

    for i in range(10):
        # Laden der csv daten in eine X sparse und eine y dense Matrix
        X, y = import_to_np(path=csv_dir, file="batch{}.csv".format(i+1))

        # Erzeugen einer vollständigen Matrix aus X und y. Konvertiere X vorher zu einer dense Matrix
        full_array= (np.c_[y.T, csr_matrix(X).toarray()])

        # Erzeugen eines Panda DataFrame Objektes aus der vollständigen dense Matrix
        df = pd.DataFrame(full_array, index=None)

        # Setze überischtlichen Header
        df = df.set_axis(header, axis=1, inplace=False)

        # Ersetze numerische durch textbasierte Klassenlabels
        for j in range(1,19):
            df.target[df.target == j] = gas[j-1]

        # Setze dtype 'category' für die Indexspalte
        df.target.astype('category')

        # Speicher DataFrame als csv Datei ab
        df.to_csv(df_original_dir + "batch{}.csv".format(i), header=True, index=None)
  
    df_modified_dir = cwd + "/input/modified_df/"

    if not os.path.exists(df_modified_dir):
        os.makedirs(df_modified_dir)
        
        
        # Folgende For-Schleife iteriert über alle Batches
    for i in range(10):
        # Lese csv Datei als Pandas DataFrame-Objekt ein
        df = pd.read_csv(df_original_dir + "batch{0}.csv".format(i), header=0, encoding='ascii', engine='python')

        # Erstelle eine Kopie
        df_copy = df.copy(deep=True)

        # Ersetze 0.1% der Daten in NaN
        num_missing = int(df_copy.shape[0] * df_copy.shape[1] * 0.001)

        # Ereuge eine Liste aus Tuples mit zufälligen Indizes
        indices = [(row,col) for row in range(df_copy.shape[0]) for col in range(1, df_copy.shape[1])]

        # Nehme diese Indizes und setze NaN im DataFrame-Objekt
        for row, col in random.sample(indices, num_missing):
            df_copy.iat[row, col] = np.nan

        # Lege ein Rauschen auf die Daten, erzeuge zunächst wieder eine Kopie noise
        noise = df_copy.copy(deep=True)

        # Da das Rauschen am Ende wieder drauf addiert werden muss, müssen die Indizes gelöscht werden
        noise.target=''

        # Erzeuge ein Rauschen aus einer Normalverteilung mit mu=0 und sigma=5% der std der jeweiligen Spalte
        for j in range(1,df_copy.shape[1]):
            noise.iloc[:,j] = np.random.normal(0, df_copy.iloc[:,j].std() * 0.1, df_copy.iloc[:,j].shape)

        # Addiere das Rauschen auf die Originaldaten
        res = df_copy + noise

        # Speichere verzerrte Daten ab
        res.to_csv(df_modified_dir + "batch{}.csv".format(i), header=True, index=None)
        
        
        
        
    
    all_files_mod = glob.glob(cwd + r"/input/modified_df/*.csv")
    print(all_files_mod)
    all_files_original = glob.glob(cwd + r"/input/original_df/*.csv")
    
    df_modified = get_dataset(all_files_mod) 
    df_original = get_dataset(all_files_original) 
    return df_modified, df_original, all_files_mod, all_files_original


def get_dataset(files, limit=None):
    li = []
    
    if (limit):
        print("hello")

    for filename in files:
        #while(limit):
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)
