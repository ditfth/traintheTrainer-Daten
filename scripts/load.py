import os
import glob
cwd = os.getcwd()

print('test')
import pandas as pd
def load(limit=None):

	all_files_mod = glob.glob(cwd + r"/input/modified_df/*.csv")
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