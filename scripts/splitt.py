import matplotlib.pyplot as plt
import pandas as pd


def show(df_modified):
	fig = plt.figure(figsize=(23.04,12.96))
	plt.plot(df_modified[df_modified['target'] == 'Ethanol low'].iloc[:,1], marker='o', linestyle='None', alpha=1, label='Ethanol low')
	plt.plot(df_modified[df_modified['target'] == 'Ethanol medium'].iloc[:,1], marker='o', linestyle='None', alpha=1, label='Ethanol medium')
	plt.plot(df_modified[df_modified['target'] == 'Ethanol high'].iloc[:,1], marker='o', linestyle='None', alpha=1, label='Ethanol high')

	plt.title('DR (Sensor 1) für Ethanol', fontsize=30)
	plt.xlabel('Messung')
	plt.ylabel('DR')

	# Instanzen der batches aufsummiert
	sum_time=[0, 445, 1689, 3275, 3436, 3633, 5933, 9546, 9840, 10310, 13910]

	for batch in sum_time:
		plt.axvline(x=batch, linewidth=1, c='r')


	plt.legend(loc='best', fontsize=30)
	plt.show()
	
def get_dataset(files, limit=None):
    li = []
    
    if (limit):
        print("debug")

    for filename in files:
        #while(limit):
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)	
	
def splitt(all_files_mod,verhaeltnis,vh):
	verhaeltnis = int(verhaeltnis/10)
	df_training = get_dataset(all_files_mod[:verhaeltnis]) 
	df_test = get_dataset(all_files_mod[verhaeltnis:]) 
	return df_training,df_test

