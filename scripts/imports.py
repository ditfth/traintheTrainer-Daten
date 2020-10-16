#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
#matplotlib inline
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

# 

# Bilder in Jupyter Notebook einfügen.
from IPython.display import Image

# Vergerben von In[] and Out[] auf der linken Seite jeder Zelle
from IPython.core.display import display,HTML
display(HTML('<style>.prompt{width: 0px; min-width: 0px; visibility: collapse}</style>'))