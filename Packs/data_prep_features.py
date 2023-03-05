import pandas as pd
import numpy as np
import ast
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def spam_ticket(df):
    if any(df[(df.part_type.isin(["note"]))&(df.entity_type=="Agent-Freehand")]['chat_text'].str.contains("spam",case=False,na=False)):
        return True
    return False
def duplicate_ticket(df):
    
    if any(df[(df.part_type.isin(["note"]))&(df.entity_type=="Agent-Freehand")]['chat_text'].str.contains("duplicate ticket",case=False,na=False)):
        return True
    return False
def abandoned_ticket(df):
    def is_consicutive(df):
        cust_adhoc=df[df.entity_type=="Customer"]
        consecutive=[]
        prev_cust_seq=0
        for indx,cust_seq in enumerate(cust_adhoc.sequence):
            if indx==0:
                prev_cust_seq=cust_seq

            else:
                if cust_seq-prev_cust_seq>1:
                    consecutive.append(False)
                else:
                    consecutive.append(True)
            prev_cust_seq=cust_seq
        if len(consecutive)!=sum(consecutive):
            return False
        else:
            return True
    if len(df[df.entity_type=="Customer"]) < 2:
        if len(df[df.entity_type=="Customer"])<1:
            return True
        else:
            cust_seq=df[df.entity_type=="Customer"].sequence.iloc[-1]
            sol_seq=None
            if df.chat_text.str.contains("#solutiondelivered|#helpneededl1l2",na=False,case=False).any():
                sol_seq=df[df.chat_text.str.contains("#solutiondelivered|#helpneededl1l2",na=False,case=False)].sequence.iloc[-1]
            if (sol_seq!=None):
                if (sol_seq<cust_seq+2):
                    return False
                else:
                    return True
            else:
                return True
    
    else:
        initial_seq = df[df.entity_type=="Customer"].sequence.iloc[0]
        end_seq = df[df.entity_type=="Customer"].sequence.iloc[-1]
        adhoc = df[(df.sequence>=initial_seq)&(df.sequence<=end_seq)]
        is_cons=is_consicutive(df)
#         tot_time = (adhoc[adhoc.entity_type=="Customer"].date_time.iloc[-1]-adhoc[adhoc.entity_type=="Customer"].date_time.iloc[0]).total_seconds()/3600
        if ((len(df[df.entity_type=="Customer"]) <3)) or (is_cons):
            if (df.chat_text.str.contains("#solutiondelivered|#helpneededl1l2|#HelpneededL1toL2",na=False,case=False).any()) & (not is_cons):
                return False
            else:
                #print(1,len(df[df.entity_type=="Customer"]),tot_time)
                return True
        else:
            
            return False 
def self_resolve(df):
    def self_resolution(session_chat):
        result= ""
        tag = session_chat['tags'].iloc[0]
        
        if type(tag)!=list:
            while type(tag)==str:
                tag = ast.literal_eval(tag)
            
            for row in tag:
                if row['name'] == 'SG-Solution - Resolved by Customer':
                    return True
            return False
        else:
            for row in tag:
                if row['name'] == 'SG-Solution - Resolved by Customer':
                    return True
            return False
    result=self_resolution(df)
    baseline="(just|i|we)(?:.*?)figure?(?:.*?)||(?:i|we) got the solution||(?:i|we?)already setup||problem solve*||(?:i|we) found(?:.*?)answer||(?:i|we) got it (?:.*?)||(?:i|we)(?:am|are|have|has) fix*||able (?:.*?) figure it"
    
    custPings=df[df.entity_type=="Customer"]
    custPings=custPings[custPings.chat_text.str.contains(r"^((?!can|how|what|why|when|may|who|where).)*$",na=False,case=False)]
    resolve=list()
    for base in tqdm(baseline.split("||")):
        resolve.append(any(custPings.chat_text.str.contains(base,na=False,case=False)))
    
    if any(resolve)==True and result==True :
        return True
    return False            
def sutherland_agent(df,agent_list):
    if df.entity_email.isin(agent_list).any():
        return 1
    else:
        return 0
def data_sequencing(df):
    
    dat=[]
    if df.columns.str.contains("sequence",na=False,case=False).any():
        df.drop("sequence",axis=1,inplace=True)
    for i in tqdm(df.session_id.unique()):
        test=df[df.session_id==i]
        test["date_time"]=pd.to_datetime(test["date_time"])
        test=test.sort_values(by="date_time")
        sequence=[i for i in range(1,len(test)+1)]
        test['sequence']=sequence
        dat.append(test)
    df2=pd.concat(dat)
    return df2
def data_correction(df,var):
    
    if df[[var]].shape[1]>1:
        df[var]=df[var].iloc[:,1]
    
    return df
