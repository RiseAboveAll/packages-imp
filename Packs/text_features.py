import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import numpy as np
import math
import sys
import ast
sys.path.insert(1,r'C:\Users\birhiman\Documents\Zenoti-Files\csat-dsat\New - Approach\CSAT-DSAT-Refinement\scripts')
sys.path.insert(2,r'C:\Users\birhiman\Documents\Zenoti-Files\csat-dsat\New - Approach\CSAT-DSAT-Refinement\utils')
import re
from similarityfunction import *
from path import *
from read_write_data import *
from data_prep_features import *
def cust_pings(df):
    #df=df[~(df.chat_text.isna()) or ~(df.chat_text==None) or ~(df.chat_text=="None") or ~(df.chat_text=="")]
    print(df.session_id.unique())
    cust_pings=[]
    if abandoned_ticket(df):
        return cust_pings.append([])
    else:
        if df[(df.part_type.isin(["note","note_and_reopen"]))].chat_text.str.contains(r"#solutiondelivered",na=False,case=False).any():
            seq_list=df[(df.part_type.isin(["note","note_and_reopen"]))&(df.chat_text.str.contains(r"#solutiondelivered",na=False,case=False))].sequence.tolist()
            print(seq_list)
            last_seq=seq_list[-1]
            cust_adhoc=df[(df.entity_type=="Customer")&(df.sequence>last_seq-3)]
            cust_pings.append(cust_adhoc.chat_text.values.tolist())
            return cust_pings
        else:
            cust_adhoc=df[(df.entity_type=="Customer")].iloc[-3:,:]
            cust_pings.append(cust_adhoc.chat_text.values.tolist())
            return cust_pings
def is_positive(df):
    if abandoned_ticket(df):
        return -1
    else:
        kpi_df=load_data(pa_cluster_path,r'\kpi_df_csat.xlsx')
        baseline=ast.literal_eval(kpi_df[kpi_df.sub_type=="CSAT_CUSTOMER"].processed_baseline.iloc[0])
        Customer=baseline['Customer']
        sentences=cust_pings(df)
        if len(sentences[0])>0:
            result=get_similarity(baselines=Customer,sentences=sentences[0],threshold=.75,debug=False).any()
            result_arr=get_similarity(baselines=Customer,sentences=sentences[0],threshold=.75,debug=False)
            indx_=np.where(result_arr==True)[0].tolist()
            if len(indx_)>0:
                reg_list=[]
                for indx in indx_:
                    
                    for i in sentences[0][indx].split(','):
                        if re.search(r"^what|^how|^when|^why|^where|unfortunately|unfortunate|check|sorry|unable|hassle",str(i).lower())==None:
                            reg_list.append(False)
                        else:
                            reg_list.append(True)
            else:
                reg_list=[]
#             if df.entity_type.isin(["Customer"]).any():
#                 initial_date=df[df.entity_type=="Customer"].date_time.iloc[0]
#             elif df[df.part_type.isin(["comment","open"])].entity_type.isin(["Agent-Freehand"]).any():
#                 initial_date=df[(df.entity_type=="Agent-Freehand")&(df.part_type.isin(["open","comment"]))].date_time.iloc[0]
#             else:
#                 initial_date=df.date_time.iloc[0]
#             if df.part_type.isin(["close"]).any():
#                 close_ticket_date=df[df.part_type=="close"].date_time.iloc[-1]
#             elif df.part_type.isin(["comment"]).any():
#                 close_ticket_date=df[df.part_type=="comment"].date_time.iloc[-1]
#             else:
#                 close_ticket_date=df.date_time.iloc[-1]
#             days=(close_ticket_date-initial_date).days
            if (result == True) and (sum(reg_list)<1) :
                return 1 #Positive reply
            else:
                return 0
        else:
            return -1

def customer_ping_last(ping,n):
    n = n/100
    final_ping = np.NaN
    data1 = ping.sort_values('sequence')
    users = data1
    users=users[users.entity_type=="Customer"]
    ping_len = len(users[users.entity_type=="Customer"]['chat_text'].values)
    if ping_len>0:
        range_list = math.ceil(ping_len * n)
        ping = users.chat_text.values[-(range_list):]
        if len(ping)>0:
            final_ping = '.'.join(ping)
        else:
            final_ping="empty"
    else:
        final_ping="empty"
    return final_ping
def last_customer_ping(ping,n):
    n = n/100
    final_ping = np.NaN
    data1 = ping.sort_values('sequence')
    users = data1
    users=users[users.entity_type=="Customer"]
    ping_len = len(users[users.entity_type=="Customer"]['chat_text'].values)
    if ping_len>0:
        range_list = math.ceil(ping_len * n)
        ping = users.chat_text.iloc[-(range_list):]
        if len(ping)>0:
            final_ping = ping
        else:
            pass
    else:
        pass
    return final_ping
def sentiment_analysis(ping):
    sid = SentimentIntensityAnalyzer()
    scores=sid.polarity_scores(ping)
    if scores["neg"] > scores["pos"]:
        return 0
    else:
        return 1