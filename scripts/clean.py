import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
cwd = os.getcwd()

#num_features = ['DR_1', 'DR_2', 'DR_3', 'DR_4']
target = ['target']



class RollingMean(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=3):
        self._window = window_size
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df_temp = pd.DataFrame(X)
        return df_temp.rolling(window=3, min_periods=1).mean().to_numpy()
        

class OutlierDetection(BaseEstimator, TransformerMixin):
    def __init__(self, m_std):
        self.m_std = m_std
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        mask = np.abs((X - X.mean(0)) / X.std(0)) > self.m_std  
        return np.where(mask, np.median(X, axis=0), X)
        
        
        
        
# Sorter: Fügt eine Spalte am Ende von X  mit numerischen Werten für die Gase hinzu unabhängig Ihrer Konzentration hinzu 
class GasLabel(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        arr = np.zeros(shape=(X.shape[0],1)) 
        for gas in range(6):
            for concentration in range(3):
                #print('gas'+str(gas)+'concentration'+str(concentration)+'X')
                arr[X[:,gas * 3 + concentration] == 1] = gas + 1
        return np.c_[X, arr]
        


def clean(df_training,df_test,num_features):
    num_pipeline = Pipeline([
        ('mean_imputer', SimpleImputer(strategy='median')),
        ('outlier_detection', OutlierDetection(std=3)),
        ('rolling_mean', RollingMean(3)),
        ('scaler', MinMaxScaler(feature_range=(0,1))),
    ])

    target_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder(sparse=False, categories='auto')),
        ('sorter', GasLabel()),
    ])

    full_pipeline = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('target', target_pipeline, target)],
        remainder='drop')
        
        
    np_train_prepared = full_pipeline.fit_transform(df_training)
    np_test_prepared = full_pipeline.transform(df_test)
    print(np_train_prepared[:2])
    print(np.isnan(np.min(np_train_prepared)), np.isnan(np.min(np_test_prepared)))
    print(full_pipeline.transformers_[1][1].steps[1][1].get_feature_names()[10] )#16 toluol low
    return np_train_prepared, np_test_prepared
    
def safe(np_train_prepared,np_test_prepared):
    csv_output_dir = cwd + "/output/csv/"
    pickle_output_dir = cwd + "/output/pickle/"

    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)
        
    if not os.path.exists(pickle_output_dir):
        os.makedirs(pickle_output_dir)
        
    pd.DataFrame(np_train_prepared).to_csv(csv_output_dir + "cleaned_train_data.csv", header=True, index=None)
    pd.DataFrame(np_test_prepared).to_csv(csv_output_dir + "cleaned_test_data.csv", header=True, index=None)

    pd.DataFrame(np_train_prepared).to_pickle(pickle_output_dir + "cleaned_train_data.pkl")
    pd.DataFrame(np_test_prepared).to_pickle(pickle_output_dir + "cleaned_test_data.pkl")
    print( 'Data saved in'+ csv_output_dir)
    

