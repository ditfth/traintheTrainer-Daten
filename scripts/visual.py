import sys

# Notwendige Pakete können direkt über Jupyter installiert werden
#!{sys.executable} -m pip install package_name

# Handhabung von Vektoren und Matrizen
import numpy as np

# Vollständige Array-Asuagabe in Jupyter Notebook
np.set_printoptions(threshold=sys.maxsize)

# Auf lokale Verzeichnisse zuzugreifen
import os

# Hole aktuelles Arbeitsverzeihcnis
cwd = os.getcwd()

# Handhabung von Pfadnamen
import glob

# Für mathematische Formeln
import math

# Shapiro-Wilks Test, Kurvenfit, Faktor
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import factorial

# Ein Paket für die Datenanalyse
import pandas as pd

# Pandas Streumatrixdiagramm
from pandas.plotting import scatter_matrix

# Analysiere Pandas DataFrames
import pandas_profiling as pp

# Verschiedene Scikit-Learn Pakete
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Um schöne Plots zu erhalten
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.patches import Rectangle

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=14)

# Für schöne Boxdiagramme (baut auf PyPlot's bxplt auf)
import seaborn as sns
pd.options.mode.chained_assignment = None


def methode1(n):
    m = 1+ math.log2(n)
    return int(m)

def methode2(n, sigma):
    m = (3.49 * sigma) / math.pow(n,1./3)
    return float(m)

def methode3(n, q_25, q_75):
    m = (2 * (q_75 - q_25)) / math.pow(n,1./3)
    return float(m)


def MinMaxTrafo(arr):
    return np.interp(arr, (arr.min(axis=0), arr.max(axis=0)), (0,+1))



	
def histo(np_train_prepared,bin):
	n_gas = 6

	gas_names = ['Acetaldyh', 'Aceton', 'Ammoniak', 'Ethanol', 'Ethen', 'Toluol']
	labels = ['low', 'medium', 'high']

	X_gas = []

	for i in range(n_gas):
		a1 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,0:4] # DR Messerwerte
		a2 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,22] # Textlabel
		a3 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,i*3+4:i*3+7] * (i+1) # Zugehörige OneHot-Vektoren
		X_gas.append(np.c_[np.c_[a2,np.roll(a3,+0,axis=1)],a1])
		
	X_gas[0][:1]
	plt.clf()

	fig = plt.figure(figsize=(23.04,12.96))
	plt.subplots_adjust(hspace=0.35)

	plt.subplot(1,2,1)
	plt.hist(X_gas[2][:,4], bins = bin)
	plt.xlabel('DR Sensor 1')
	plt.title('Bin = '+str(bin))
	plt.ylabel('Häufigkeit')



	plt.suptitle('Ammoniak', fontsize=28)

	plt.show()


def histo2(np_train_prepared):
	n_gas = 6

	gas_names = ['Acetaldyh', 'Aceton', 'Ammoniak', 'Ethanol', 'Ethen', 'Toluol']
	labels = ['low', 'medium', 'high']

	X_gas = []

	for i in range(n_gas):
		a1 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,0:4] # DR Messerwerte
		a2 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,22] # Textlabel
		a3 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,i*3+4:i*3+7] * (i+1) # Zugehörige OneHot-Vektoren
		X_gas.append(np.c_[np.c_[a2,np.roll(a3,+0,axis=1)],a1])
		
	X_gas[0][:1]
	
	anzahl = X_gas[2].shape[0]
	std = X_gas[2][:,4].std()
	std2 = (X_gas[2][:,4].max() - X_gas[2][:,4].min()) / 4.
	Q25 = np.percentile(X_gas[2][:,4], 25)
	Q75 = np.percentile(X_gas[2][:,4], 75)

	plt.clf()
	fig = plt.figure(figsize=(23.04,12.96))

	plt.subplot(3,2,1)
	bins = methode1(anzahl)
	plt.hist(X_gas[2][:,4], bins = bins)
	plt.xlabel('DR Sensor 1')
	plt.ylabel('Häufigkeit')
	plt.title('Methode 1: Bin = {}'.format(bins))

	plt.subplot(3,2,2)
	bins = methode2(anzahl, std2)
	plt.hist(X_gas[2][:,4], bins = int(1 / bins))
	plt.xlabel('DR Sensor 1')
	plt.ylabel('Häufigkeit')
	plt.title('Methode 2: Bin = {}'.format(int(1 / bins)))

	plt.subplot(3,2,3)
	bins = methode3(anzahl, Q25, Q75)
	plt.hist(X_gas[2][:,4], bins = int(1 / bins))
	plt.xlabel('DR Sensor 1')
	plt.ylabel('Häufigkeit')
	plt.title('Methode 3: Bin = {}'.format(int(1 / bins)))

	plt.subplot(3,2,4)
	bins = [0.0125, 0.025, 0.0375,  0.05, 0.0625, 0.0750, 0.0875,  0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0]
	plt.hist(X_gas[2][:,4], bins = bins, density=True)
	plt.xlabel('DR Sensor 1')
	plt.ylabel('Häufigkeit (normiert)')
	plt.title('Methode 4: Bin = {}'.format('dynamisch'))

	plt.suptitle('Ammoniak', fontsize=28)
	plt.show()
	
def histo1d(np_train_prepared):
	n_gas = 6

	gas_names = ['Acetaldyh', 'Aceton', 'Ammoniak', 'Ethanol', 'Ethen', 'Toluol']
	labels = ['low', 'medium', 'high']

	X_gas = []

	for i in range(n_gas):
		a1 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,0:4] # DR Messerwerte
		a2 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,22] # Textlabel
		a3 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,i*3+4:i*3+7] * (i+1) # Zugehörige OneHot-Vektoren
		X_gas.append(np.c_[np.c_[a2,np.roll(a3,+0,axis=1)],a1])
		
	X_gas[0][:1]
	
	
	plt.clf()
	fig = plt.figure(figsize=(23.04,12.96))

	bins = np.arange(len(gas_names)+1) + 0.5

	color = ['c', 'm', 'y']

	for gas in range(n_gas):
		data = []
		for concentration in range(3):
			data.append(X_gas[gas][:,0][X_gas[gas][:,concentration+1] == gas+1])
		plt.hist(data, bins='auto', color=color, label=labels)


	handles = [Rectangle((0,0),1,1,color=c) for c in color]
	labels= ['low','medium', 'high']
	plt.legend(handles, labels, fontsize=30)

	plt.title("Konzentrationen für verschiedene Gase")
	plt.xlabel("Gase", fontsize=20)
	plt.ylabel('Häufigkeit')
	plt.xticks(bins+0.5, gas_names)
	plt.xlim([0.5,6.5])

	plt.show()
	
def histo1da(np_train_prepared):
	n_gas = 6

	gas_names = ['Acetaldyh', 'Aceton', 'Ammoniak', 'Ethanol', 'Ethen', 'Toluol']
	labels = ['low', 'medium', 'high']

	X_gas = []

	for i in range(n_gas):
		a1 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,0:4] # DR Messerwerte
		a2 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,22] # Textlabel
		a3 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,i*3+4:i*3+7] * (i+1) # Zugehörige OneHot-Vektoren
		X_gas.append(np.c_[np.c_[a2,np.roll(a3,+0,axis=1)],a1])
		
	X_gas[0][:1]	
	plt.clf()

	fig = plt.figure(figsize=(23.04,12.96))
	plt.subplots_adjust(hspace=0.25)

	for gas in range(n_gas):
		plt.subplot(3,2,gas+1)
		plt.hist(X_gas[gas][:,4], bins=20)
		plt.title(gas_names[gas],fontsize=20)
		plt.ylabel('Häufigkeit')
	plt.suptitle("DR für Sensor 1", fontsize=30)

	plt.show()
	
def sym(np_train_prepared):
	n_gas = 6

	gas_names = ['Acetaldyh', 'Aceton', 'Ammoniak', 'Ethanol', 'Ethen', 'Toluol']
	labels = ['low', 'medium', 'high']

	X_gas = []

	for i in range(n_gas):
		a1 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,0:4] # DR Messerwerte
		a2 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,22] # Textlabel
		a3 = np_train_prepared[np_train_prepared[:,22] == int(i+1)][:,i*3+4:i*3+7] * (i+1) # Zugehörige OneHot-Vektoren
		X_gas.append(np.c_[np.c_[a2,np.roll(a3,+0,axis=1)],a1])
		
	X_gas[0][:1]
	
	
	original = X_gas[4][:,4][X_gas[4][:,4] != 0]
	square = MinMaxTrafo(np.sqrt(original))
	rezi = MinMaxTrafo(np.reciprocal(original))
	log = MinMaxTrafo(np.log(original))
	boxcox = MinMaxTrafo(stats.boxcox(original)[0])
	
	plt.clf()
	fig = plt.figure(figsize=(23.04,12.96))

	kwards = dict(alpha=0.7, bins=70)

	plt.hist(original, **kwards, label='Original')
	plt.hist(square, **kwards, label='Quadratwurzel')
	plt.hist(rezi, **kwards, label='Reziprok')
	plt.hist(log, **kwards, label='Logarithmus')
	plt.hist(boxcox, **kwards, label='Box-Cox')

	plt.legend(loc='best', fontsize=30)
	plt.title('Transformation Schief Verteilter Daten', fontsize=30)
	plt.ylabel('Häufigkeit')
	plt.xlabel('DR')

	plt.show()
	
	conc_array = np.concatenate(
    (original.reshape(-1,1),
    square.reshape(-1,1),
    rezi.reshape(-1,1),
    log.reshape(-1,1),
    boxcox.reshape(-1,1)),
    axis=1)

	(pd.DataFrame(conc_array)).skew(axis=0)
	
	
	
	
def multhisto(X_gas):	
	plt.clf()
	fig = plt.figure(figsize=(23.04,12.96))

	colors = ['g', 'b', 'r', 'y']
	kwargs = dict(alpha=0.5, bins=100)
	for color in range(len(colors)):
		plt.hist(X_gas[4][:,4+color], **kwargs, color=colors[color], label='Sensor {}'.format(color+1))

	plt.title('Ethen', fontsize=30)
	plt.ylabel('Häufigkeit')
	plt.xlabel('DR')
	plt.legend(loc='best', fontsize=30)

	plt.show()
	
	
def histo2d(X_gas):	
	plt.clf()

	fig = plt.figure(figsize=(23.04,12.96))

	gas = X_gas[0]
	plt.hexbin(gas[:,6],gas[:,7], cmap='inferno')

	plt.title('Acetaldyhyd DR', fontsize=30)
	plt.xlabel('Sensor 3')
	plt.ylabel('Sensor 4')

	plt.show()
	
	
def time(X_gas):	
	plt.clf()

	fig = plt.figure(figsize=(23.04,12.96))

	conc_ethanol = np.empty(shape=(X_gas[3].shape[0],)).astype(str)
	conc_ethanol[X_gas[3][:,2] == 4] = 'medium'
	conc_ethanol[X_gas[3][:,1] == 4] = 'low'
	conc_ethanol[X_gas[3][:,3] == 4] = 'high'

	plt.plot(conc_ethanol[:100], label='Ethanol', marker='o')

	plt.title('Konzentrationen für Ethanol', fontsize=30)
	plt.xlabel('Messpunkt')
	plt.ylabel('Konzentration')

	plt.show()
	
	
def time2(X_gas):	

	plt.clf()
	fig = plt.figure(figsize=(23.04,12.96))

	plt.plot(X_gas[3][:,4], alpha=1, label='Sensor 1')
	plt.plot(X_gas[3][:,5], alpha=0.7, label='Sensor 2')
	plt.plot(X_gas[3][:,6], alpha=0.5, label='Sensor 3')

	plt.title('DR für Ethanol der ersten drei Sensoren', fontsize=30)
	plt.xlabel('Messpunkt')
	plt.ylabel('DR')

	plt.legend(loc='best', fontsize=30)

	plt.show()
	
	
	
def time3(X_gas):	

	plt.clf()
	fig = plt.figure(figsize=(23.04,12.96))

	plt.plot(X_gas[3][:,4], alpha=1, label='Ethanol')
	plt.plot(X_gas[1][:,4], alpha=1, label='Aceton')
	plt.plot(X_gas[0][:,4], alpha=1, label='Acetaldehyd')

	plt.title('DR_1 für Ethanol Aceton und Acetaldehyd', fontsize=30)
	plt.xlabel('Messpunkt')
	plt.ylabel('DR Sensor 1')

	plt.legend(loc='best', fontsize=30)

	plt.show()
	

def mittel(X_gas):
	mean = X_gas[4][:100,4:].mean(axis=1)
	std = X_gas[4][:100,4:].std(axis=1)
	plt.clf()
	fig = plt.figure(figsize=(23.04,12.96))

	time = np.arange(mean.shape[0])
	plt.fill_between(time, mean-2*std, mean+2*std, alpha=0.5, color='yellow', label=r'$\pm 2\sigma$')
	plt.fill_between(time, mean-std, mean+std, alpha=0.5, color='green', label=r'$\pm 1\sigma$')
	plt.plot(time, mean, color='black', label='mean')

	plt.title('Mittelwert mit Standardabweichung', fontsize=30)
	plt.xlabel('Messpunkt')
	plt.ylabel('DR')
	plt.legend(loc='best', fontsize=30)

	plt.show()
	
	
	
def streu(X_gas):	
	plt.clf()
	fig = plt.figure(figsize=(23.04,12.96))

	plt.scatter(X_gas[0][:,4], X_gas[0][:,7], label='Ethanol', c='blue')
	plt.scatter(X_gas[1][:,4], X_gas[1][:,7], label='Ethen', c='red')
	plt.scatter(X_gas[2][:,4], X_gas[2][:,7], label='Ammoniak', c='green')
	plt.scatter(X_gas[3][:,4], X_gas[3][:,7], label='Acetaldehyd', c='purple')

	plt.xlim(0,1.1)
	plt.ylim(0,1)

	plt.title('Streudiagramm für Sensor 1 und 3 für verschiedene Gase', fontsize=30)
	plt.xlabel('DR für Sensor 1')
	plt.ylabel('DR für Sensor 3')
	plt.legend(loc='best', fontsize='x-large')


	plt.show()
	


def streumat(X_gas):
	df = pd.DataFrame(X_gas[4])
	attributes = [4, 5, 6, 7] # Für die Sensoren eins bis vier
	pandas_plot = scatter_matrix(df[attributes], figsize=(20,20), diagonal='kde')

def box(np_train_prepared,X_gas,gas_names,n_gas):
	data_np = np.c_[np.c_[np_train_prepared[:,0], np_train_prepared[:,2]], np_train_prepared[:,22]]
	data_dict = {'DR_1' : data_np[:,0],'DR_3' : data_np[:,1], 'target' : data_np[:,2]}
	df = pd.DataFrame(data_dict)

	for gas in range(n_gas):
		df['target'][df['target'] == int(gas+1)] = gas_names[gas]
		
		
	plt.clf()
	fig = plt.figure(figsize=(23.04,12.96))
	sns.boxplot(x='target', y='DR_1', data=df,  width=0.5, palette='colorblind', fliersize=5, whis=3)
	
	
def streuhist(X_gas):
	# Definiere Layout
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.65
	bottom_h = left_h = left + width + 0.02

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]
	rect_histy = [left_h, bottom, 0.2, height]

	# Definiere Größe
	plt.figure(1, figsize=(20, 20))
	axScatter = plt.axes(rect_scatter)
	axHistx = plt.axes(rect_histx)
	axHisty = plt.axes(rect_histy)

	label = ['low', 'medium', 'high']
	colors = ['r', 'g', 'b']

	sensor_2 = [
		X_gas[1][:,5][X_gas[1][:,1] == 2],
		X_gas[1][:,5][X_gas[1][:,2] == 2], 
		X_gas[1][:,5][X_gas[1][:,3] == 2]
	]
	sensor_3 = [
		X_gas[1][:,6][X_gas[1][:,1] == 2], 
		X_gas[1][:,6][X_gas[1][:,2] == 2], 
		X_gas[1][:,6][X_gas[1][:,3] == 2]
	]

	# Definiere die Binbreite von Hand
	binwidth = 0.025
	lim = (int(1/binwidth) + 1) * binwidth
	bins = np.arange(-lim, lim + binwidth, binwidth)

	kwargs = dict(edgecolors='c', alpha=0.3, edgecolor='k', s=50)
	kwargsh = dict(bins=bins, alpha=0.5, density=True)

	for color in range(len(colors)):
		axScatter.scatter(sensor_2[color], sensor_3[color],  **kwargs, c=colors[color], label=label[color])
		axHistx.hist(sensor_2[color], **kwargsh, color=colors[color])
		axHisty.hist(sensor_3[color], **kwargsh, color=colors[color], orientation='horizontal')
		
	axScatter.set_xlim(.0,1.)
	axScatter.set_ylim(.0,1.)
	axHistx.set_xlim(axScatter.get_xlim())
	axHisty.set_ylim(axScatter.get_ylim())

	axScatter.set_xlabel('DR Sensor 1')
	axScatter.set_ylabel('DR Sensor 2')
	axScatter.legend(loc='best', fontsize=30)

	plt.show()