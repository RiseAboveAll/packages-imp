import pandas as pd

def load_data(path,filename):
    ext=filename.split('.')[-1]
    if ext=="pkl" or ext=="pickle":
        df=pd.read_pickle(path+filename)
        return df
    elif ext=="csv":
        df=pd.read_csv(path+filename)
        return df
    elif ext=="xlsx":
        df=pd.read_excel(path+filename)
        return df
    else:
        print("data format should be in form of pickle or csv or xlsx")
    