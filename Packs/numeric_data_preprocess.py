import pandas as pd
import numpy as np
import pickle
from path import *
def percentile(df,var,perc):
    high_perc=df[var].quantile(perc)
    low_perc=df[var].quantile(1-perc)
    return (low_perc,high_perc)
def outlier_treatment(df,var,perc):
    indx=df[df[var]<percentile(df,var,perc)[0]].index.tolist()
    df.loc[indx,var]=df[var].median()
    indx=df[df[var]>percentile(df,var,perc)[1]].index.tolist()
    df.loc[indx,var]=percentile(df,var,perc)[1]
    return df
class RobustScaler:
    def __init__(self,outlier_config={},config=None):
        
        self.outlier_config=outlier_config
        
        self.config=config
        self.high_perc=None
        self.low_perc=None
    def fit_transform(self,df,file_name='scaler.pkl'):
        self.data=df
        for key,value in self.config.items():
            self.high_perc=self.data[key].quantile(value)
            self.low_perc=self.data[key].quantile(1-value)
            self.outlier_config[key]=[self.high_perc,self.low_perc]
            self.data[key]=self.data[key].map(lambda x: (x-self.low_perc)/(self.high_perc-self.low_perc))
        filehandler = open(pa_cluster_path+f'\{file_name}', 'wb')
        pickle.dump(self.outlier_config, filehandler)
        filehandler.close()
        return self.data
    def transform(self,val,scaler_pickle='scaler.pkl'):
        if (len(self.outlier_config)<1) or (self.outlier_config==None):
            scaler=pa_cluster_path+f'\{scaler_pickle}'
            pickle_in = open(scaler,"rb")
            self.outlier_config = pickle.load(pickle_in)
            
            
        for key,_ in self.config.items():
            high_perc,low_perc=self.outlier_config[key]
            val[key]=val[key].map(lambda x: (x-low_perc)/(high_perc-low_perc))
        return val
    def inverse_transform(self,df):
        if len(self.outlier_config)<1:
            scaler=pa_cluster_path+r'\scaler.pkl'
            pickle_in = open(scaler,"rb")
            self.outlier_config = pickle.load(pickle_in)
        for key,_ in self.config.items():
            high_perc,low_perc=self.outlier_config[key]
            df[key]=df[key].map(lambda x: (x*(high_perc-low_perc))+low_perc)
        return df
    