import pandas as pd
import numpy as np

def PA_Encoding(df,tags_dict:dict):
    df["PA_Cluster"]=None
    gp1=tags_dict["Appointment"]
    gp2=tags_dict["Billing"]
    gp3=tags_dict["Business Intelligence"]
    gp4=tags_dict["Business Management"]
    gp5=tags_dict["Marketing and Sales"]
    gp6=tags_dict["Misc"]
    gp7=tags_dict["Mobile Application"]
    gp8=tags_dict["Payment"]
    df["PA_Tags"]=df["PA_Tags"].map(lambda x:str(x).lower())
    df["PA_Tags"].replace("none",np.nan,inplace=True)
    df["PA_Cluster"]=df["PA_Tags"].map(lambda x: "Appointment" if str(x) in [str(i).lower() for i in gp1] else "Billing" if str(x) in [str(i).lower() for i in gp2] else "Business Intelligence" if str(x) in [str(i).lower() for i in gp3] else "Business Management" if str(x) in [str(i).lower() for i in gp4] else "Marketing & Sales" if str(x) in [str(i).lower() for i in gp5] else "Misc" if str(x) in [str(i).lower() for i in gp6] else "Mobile Application" if str(x) in [str(i).lower() for i in gp7] else "Payment" if str(x) in [str(i).lower() for i in gp8] else np.nan if pd.isna(x) else "Misc" )
    return df