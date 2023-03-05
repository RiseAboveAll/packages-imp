import pandas as pd
import numpy as np
import ast
import warnings
from tqdm import tqdm
from data_prep_features import *
import sys
sys.path.insert(1,r'C:\Users\birhiman\Documents\Zenoti-Files\csat-dsat\New - Approach\CSAT-DSAT-Refinement\scripts')
from path import *
import pickle
warnings.filterwarnings('ignore')
def irt(df):
    if (df[df.part_type.isin(['open',"comment"])].entity_type.isin(["Agent-Freehand"]).any()) & (df.entity_type.isin(["Customer"]).any()):
        init_cust_seq=df[df.entity_type=="Customer"].sequence.iloc[0]
        adhoc=df[df.sequence>=init_cust_seq]
        if (adhoc[adhoc.part_type.isin(['open',"comment"])].entity_type.isin(["Agent-Freehand"]).any()):
            init_agent_seq=adhoc[(adhoc.entity_type=="Agent-Freehand")&(adhoc.part_type.isin(['open','comment']))].sequence.iloc[0]
            cust_time=pd.to_datetime(adhoc.date_time.iloc[0])
            agent_time=pd.to_datetime(adhoc[adhoc.sequence==init_agent_seq].date_time.iloc[0])
            return (agent_time-cust_time).total_seconds()/60
        else:
            return np.NaN
    else:
        return np.NaN
def cust_waiting_time(df):
    d=df[df.part_type.isin(["comment","open","close","assignment","note"])]
    d.reset_index(drop=True,inplace=True)
    if d.entity_type.isin(["Customer"]).any():
        d=d[d.index>=d[d.entity_type.isin(["Customer"])].index[0]]

        if d.entity_type.isin(["Customer"]).any() & d[d.part_type.isin(["open","close","comment"])].entity_type.isin(["Agent-Freehand"]).any() :
            d=d[~(d.chat_text.isna())]
            cust_index=d[d.entity_type=="Customer"].sequence.tolist()
            agent_index=d[d.entity_type.isin(["System","Agent-Freehand"])&(d.part_type.isin(["comment","open","close"]))].sequence.tolist()
            index_diff=[]
            for indx,cust in enumerate(cust_index):
                

                for _,agent in enumerate(agent_index):
                    
                    if indx==0:

                        index_diff.append((agent,cust,agent-cust))
                        initial_agent_index=cust
                        
                    else:

                        if agent-initial_agent_index>0:
                            index_diff.append((agent,cust,agent-cust))
                            initial_agent_index=cust
#                         else:
#                             initial_agent_index=cust
                            
            indx_diff=pd.DataFrame(index_diff,columns=["agent_indx","cust_indx","diff"])
            def is_index_lesser(a,b,c):
                if a<b:
                    return True
                else:
                    return False
            indx_diff["is_lesser"]=indx_diff.apply(lambda x: is_index_lesser(a=x["cust_indx"],b=x["agent_indx"],c=x["diff"]),axis=1)
            sub=indx_diff[indx_diff.is_lesser==True]
            if len(sub)>0:
                sub=sub.groupby("cust_indx",as_index=False)["diff"].min().merge(sub,on=["cust_indx","diff"])
                sub=sub.groupby("agent_indx",as_index=False)["diff"].max().merge(sub,on=["agent_indx","diff"])
                da={}
                di=[]
                agntIndx=[]
                custIndx=[]
                for i in sub.cust_indx.unique():
                    di.append(sub[sub.cust_indx==i]["diff"].min())
                    agntIndx.append(sub[(sub["diff"]==sub[sub.cust_indx==i]["diff"].min())&(sub.cust_indx==i)].agent_indx.iloc[0])
                    custIndx.append(i)
                    da["diff"]=di
                    da["agent_indx"]=agntIndx
                    da["cust_indx"]=custIndx
                sub=pd.DataFrame(da)

                indx=[]
                for cust,agent in zip(sub.cust_indx,sub.agent_indx):
                    indx.append((cust,agent))
                time=[]
                for i in indx:
                    cust_time=pd.to_datetime(d[d.sequence==i[0]].loc[:,'date_time'].iloc[0])
                    agnt_time=pd.to_datetime(d[d.sequence==i[1]].loc[:,'date_time'].iloc[0])
                    time.append((agnt_time-cust_time).total_seconds())
                return np.median(time)/60 #return average agent waiting time for customer reply 
            else:
                return np.NaN
        else:
            if d.entity_type.isin(["System"]).any():
                time=[]
                cust_seq=d[d.entity_type.isin(["Customer"])].sequence.iloc[0]
                syst_seq=d[d.entity_type.isin(["System"])].sequence.iloc[-1]
                if syst_seq>cust_seq:
                    cust_time=pd.to_datetime(d[d.sequence==cust_seq].loc[:,'date_time'].iloc[0])
                    syst_time=pd.to_datetime(d[d.sequence==syst_seq].loc[:,'date_time'].iloc[0])
                    time.append((syst_time-cust_time).total_seconds())
                    return np.median(time)/60
                else:
                    return np.NaN
            else:
                return np.NaN
                    
                    
                
            
    else:
        return np.NaN
def max_cust_waiting_time(df):
    d=df[df.part_type.isin(["comment","open","close","assignment","note"])]
    d.reset_index(drop=True,inplace=True)
    if d.entity_type.isin(["Customer"]).any():
        d=d[d.index>=d[d.entity_type.isin(["Customer"])].index[0]]

        if d.entity_type.isin(["Customer"]).any() & d[d.part_type.isin(["open","close","comment"])].entity_type.isin(["Agent-Freehand"]).any() :
            d=d[~(d.chat_text.isna())]
            cust_index=d[d.entity_type=="Customer"].sequence.tolist()
            agent_index=d[d.entity_type.isin(["System","Agent-Freehand"])&(d.part_type.isin(["comment","open","close"]))].sequence.tolist()
            index_diff=[]
            for indx,cust in enumerate(cust_index):
                

                for _,agent in enumerate(agent_index):
                    
                    if indx==0:

                        index_diff.append((agent,cust,agent-cust))
                        initial_agent_index=cust
                        
                    else:

                        if agent-initial_agent_index>0:
                            index_diff.append((agent,cust,agent-cust))
                            initial_agent_index=cust
                            
            indx_diff=pd.DataFrame(index_diff,columns=["agent_indx","cust_indx","diff"])
            def is_index_lesser(a,b,c):
                if a<b:
                    return True
                else:
                    return False
            indx_diff["is_lesser"]=indx_diff.apply(lambda x: is_index_lesser(a=x["cust_indx"],b=x["agent_indx"],c=x["diff"]),axis=1)
            sub=indx_diff[indx_diff.is_lesser==True]
            if len(sub)>0:
                sub=sub.groupby("cust_indx",as_index=False)["diff"].min().merge(sub,on=["cust_indx","diff"])
                sub=sub.groupby("agent_indx",as_index=False)["diff"].max().merge(sub,on=["agent_indx","diff"])
                da={}
                di=[]
                agntIndx=[]
                custIndx=[]
                for i in sub.cust_indx.unique():
                    di.append(sub[sub.cust_indx==i]["diff"].min())
                    agntIndx.append(sub[(sub["diff"]==sub[sub.cust_indx==i]["diff"].min())&(sub.cust_indx==i)].agent_indx.iloc[0])
                    custIndx.append(i)
                    da["diff"]=di
                    da["agent_indx"]=agntIndx
                    da["cust_indx"]=custIndx
                sub=pd.DataFrame(da)

                indx=[]
                for cust,agent in zip(sub.cust_indx,sub.agent_indx):
                    indx.append((cust,agent))
                time=[]
                for i in indx:
                    cust_time=pd.to_datetime(d[d.sequence==i[0]].loc[:,'date_time'].iloc[0])
                    agnt_time=pd.to_datetime(d[d.sequence==i[1]].loc[:,'date_time'].iloc[0])
                    time.append((agnt_time-cust_time).total_seconds())
                return max(time)/60 #return average agent waiting time for customer reply
            else:
                return np.NaN
        else:
            if d.entity_type.isin(["System"]).any():
                time=[]
                cust_seq=d[d.entity_type.isin(["Customer"])].sequence.iloc[0]
                syst_seq=d[d.entity_type.isin(["System"])].sequence.iloc[-1]
                if syst_seq>cust_seq:
                    cust_time=pd.to_datetime(d[d.sequence==cust_seq].loc[:,'date_time'].iloc[0])
                    syst_time=pd.to_datetime(d[d.sequence==syst_seq].loc[:,'date_time'].iloc[0])
                    time.append((syst_time-cust_time).total_seconds())
                    return max(time)/60
                else:
                    return np.NaN
            else:
                return np.NaN
    else:
        return np.NaN
def min_cust_waiting_time(df):
    d=df[df.part_type.isin(["comment","open","close","assignment","note"])]
    d.reset_index(drop=True,inplace=True)
    if d.entity_type.isin(["Customer"]).any():
        d=d[d.index>=d[d.entity_type.isin(["Customer"])].index[0]]

        if d.entity_type.isin(["Customer"]).any() & d[d.part_type.isin(["open","close","comment"])].entity_type.isin(["Agent-Freehand"]).any() :
            d=d[~(d.chat_text.isna())]
            cust_index=d[d.entity_type=="Customer"].sequence.tolist()
            agent_index=d[d.entity_type.isin(["System","Agent-Freehand"])&(d.part_type.isin(["comment","open","close"]))].sequence.tolist()
            index_diff=[]
            for indx,cust in enumerate(cust_index):
                

                for _,agent in enumerate(agent_index):
                    
                    if indx==0:

                        index_diff.append((agent,cust,agent-cust))
                        initial_agent_index=cust
                        
                    else:

                        if agent-initial_agent_index>0:
                            index_diff.append((agent,cust,agent-cust))
                            initial_agent_index=cust
                            
            indx_diff=pd.DataFrame(index_diff,columns=["agent_indx","cust_indx","diff"])
            def is_index_lesser(a,b,c):
                if a<b:
                    return True
                else:
                    return False
            indx_diff["is_lesser"]=indx_diff.apply(lambda x: is_index_lesser(a=x["cust_indx"],b=x["agent_indx"],c=x["diff"]),axis=1)
            sub=indx_diff[indx_diff.is_lesser==True]
            if len(sub)>0:
                sub=sub.groupby("cust_indx",as_index=False)["diff"].min().merge(sub,on=["cust_indx","diff"])
                sub=sub.groupby("agent_indx",as_index=False)["diff"].max().merge(sub,on=["agent_indx","diff"])
                da={}
                di=[]
                agntIndx=[]
                custIndx=[]
                for i in sub.cust_indx.unique():
                    di.append(sub[sub.cust_indx==i]["diff"].min())
                    agntIndx.append(sub[(sub["diff"]==sub[sub.cust_indx==i]["diff"].min())&(sub.cust_indx==i)].agent_indx.iloc[0])
                    custIndx.append(i)
                    da["diff"]=di
                    da["agent_indx"]=agntIndx
                    da["cust_indx"]=custIndx
                sub=pd.DataFrame(da)

                indx=[]
                for cust,agent in zip(sub.cust_indx,sub.agent_indx):
                    indx.append((cust,agent))
                time=[]
                for i in indx:
                    cust_time=pd.to_datetime(d[d.sequence==i[0]].loc[:,'date_time'].iloc[0])
                    agnt_time=pd.to_datetime(d[d.sequence==i[1]].loc[:,'date_time'].iloc[0])
                    time.append((agnt_time-cust_time).total_seconds())
                return min(time)/60 #return average agent waiting time for customer reply
            else:
                return np.NaN
        else:
            if d.entity_type.isin(["System"]).any():
                time=[]
                cust_seq=d[d.entity_type.isin(["Customer"])].sequence.iloc[0]
                syst_seq=d[d.entity_type.isin(["System"])].sequence.iloc[-1]
                if syst_seq>cust_seq:
                    cust_time=pd.to_datetime(d[d.sequence==cust_seq].loc[:,'date_time'].iloc[0])
                    syst_time=pd.to_datetime(d[d.sequence==syst_seq].loc[:,'date_time'].iloc[0])
                    time.append((syst_time-cust_time).total_seconds())
                    return max(time)/60
                else:
                    return np.NaN
            else:
                return np.NaN
    else:
        return np.NaN
def count_assignment(df):
    count_assign=df[df.part_type.isin(["assignment","away_mode_assignment"])].shape[0]
    return count_assign
def agent_reply_time_after_assignment(df):
    if count_assignment(df)>0:
        assign_seq=df[df.part_type.isin(["assignment","away_mode_assignment"])].sequence.tolist()
        time=[]
        for i in assign_seq:

            subdf=df[df.sequence>i]
            if subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment"]))].shape[0]>0:
                if subdf.part_type.isin(["assignment","away_mode_assignment"]).any():
                    next_assign_seq=subdf[subdf.part_type.isin(["assignment","away_mode_assignment"])].sequence.iloc[0]
                else:
                    next_assign_seq=None
                if next_assign_seq is not None:
                    sub_df=subdf[(subdf.sequence<next_assign_seq)&(subdf.entity_type=="Agent-Freehand")]
                    if len(sub_df)>0 & sub_df.part_type.isin(["open","close","comment"]).any():
                        assign_time=df[df.sequence==i].date_time.iloc[0]
                        next_agent_cmnt_time=subdf[(subdf.part_type.isin(["open","close","comment"]))&(subdf.entity_type=="Agent-Freehand")].date_time.iloc[0]
                        time.append((next_agent_cmnt_time-assign_time).total_seconds()/60)
                    else:
                        assign_seq.remove(next_assign_seq)
                        assign_time=df[df.sequence==i].date_time.iloc[0]
                        next_agent_cmnt_time=subdf[(subdf.part_type.isin(["open","close","comment"]))&(subdf.entity_type=="Agent-Freehand")].date_time.iloc[0]
                        time.append((next_agent_cmnt_time-assign_time).total_seconds()/60)
                else:
                    assign_time=df[df.sequence==i].date_time.iloc[0]
                    next_agent_cmnt_time=subdf[(subdf.part_type.isin(["open","close","comment"]))&(subdf.entity_type=="Agent-Freehand")].date_time.iloc[0]
                    time.append((next_agent_cmnt_time-assign_time).total_seconds()/60)

            else:
                time=[]
        if len(time)>0:
            return np.median(time)
        else:
            return np.NaN
    else:
        return np.NaN
        
def max_agent_reply_time_after_assignment(df):
    if count_assignment(df)>0:
        assign_seq=df[df.part_type.isin(["assignment","away_mode_assignment"])].sequence.tolist()
        time=[]
        for i in assign_seq:

            subdf=df[df.sequence>i]
            if subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment"]))].shape[0]>0:
                if subdf.part_type.isin(["assignment","away_mode_assignment"]).any():
                    next_assign_seq=subdf[subdf.part_type.isin(["assignment","away_mode_assignment"])].sequence.iloc[0]
                else:
                    next_assign_seq=None
                if next_assign_seq is not None:
                    sub_df=subdf[(subdf.sequence<next_assign_seq)&(subdf.entity_type=="Agent-Freehand")]
                    if len(sub_df)>0 & sub_df.part_type.isin(["open","close","comment"]).any():
                        assign_time=df[df.sequence==i].date_time.iloc[0]
                        next_agent_cmnt_time=subdf[(subdf.part_type.isin(["open","close","comment"]))&(subdf.entity_type=="Agent-Freehand")].date_time.iloc[0]
                        time.append((next_agent_cmnt_time-assign_time).total_seconds()/60)
                    else:
                        assign_seq.remove(next_assign_seq)
                        assign_time=df[df.sequence==i].date_time.iloc[0]
                        next_agent_cmnt_time=subdf[(subdf.part_type.isin(["open","close","comment"]))&(subdf.entity_type=="Agent-Freehand")].date_time.iloc[0]
                        time.append((next_agent_cmnt_time-assign_time).total_seconds()/60)
                else:
                    assign_time=df[df.sequence==i].date_time.iloc[0]
                    next_agent_cmnt_time=subdf[(subdf.part_type.isin(["open","close","comment"]))&(subdf.entity_type=="Agent-Freehand")].date_time.iloc[0]
                    time.append((next_agent_cmnt_time-assign_time).total_seconds()/60)
            else:
                time=[]
        if len(time)>0:
            return max(time)
        else:
            return np.NaN
    else:
        return np.NaN
def min_agent_reply_time_after_assignment(df):
    if count_assignment(df)>0:
        assign_seq=df[df.part_type.isin(["assignment","away_mode_assignment"])].sequence.tolist()
        time=[]
        for i in assign_seq:

            subdf=df[df.sequence>i]
            if subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment"]))].shape[0]>0:
                if subdf.part_type.isin(["assignment","away_mode_assignment"]).any():
                    next_assign_seq=subdf[subdf.part_type.isin(["assignment","away_mode_assignment"])].sequence.iloc[0]
                else:
                    next_assign_seq=None
                if next_assign_seq is not None:
                    sub_df=subdf[(subdf.sequence<next_assign_seq)&(subdf.entity_type=="Agent-Freehand")]
                    if len(sub_df)>0 & sub_df.part_type.isin(["open","close","comment"]).any():
                        assign_time=df[df.sequence==i].date_time.iloc[0]
                        next_agent_cmnt_time=subdf[(subdf.part_type.isin(["open","close","comment"]))&(subdf.entity_type=="Agent-Freehand")].date_time.iloc[0]
                        time.append((next_agent_cmnt_time-assign_time).total_seconds()/60)
                    else:
                        assign_seq.remove(next_assign_seq)
                        assign_time=df[df.sequence==i].date_time.iloc[0]
                        next_agent_cmnt_time=subdf[(subdf.part_type.isin(["open","close","comment"]))&(subdf.entity_type=="Agent-Freehand")].date_time.iloc[0]
                        time.append((next_agent_cmnt_time-assign_time).total_seconds()/60)
                else:
                    assign_time=df[df.sequence==i].date_time.iloc[0]
                    next_agent_cmnt_time=subdf[(subdf.part_type.isin(["open","close","comment"]))&(subdf.entity_type=="Agent-Freehand")].date_time.iloc[0]
                    time.append((next_agent_cmnt_time-assign_time).total_seconds()/60)
            else:
                time=[]
        if len(time)>0:
            return min(time)
        else:
            return np.NaN
    else:
        return np.NaN
def avg_agent_system_waiting_time(df):
        
    
    d=df[df.part_type.isin(["comment","open","close","assignment","note"])]
    d.reset_index(drop=True,inplace=True)
    if d.entity_type.isin(["Agent-Freehand","System"]).any():
        d=d[d.index>=d[d.entity_type.isin(["Agent-Freehand","System"])].index[0]]

        if d.entity_type.isin(["Customer"]).any() & d[d.part_type.isin(["open","close","comment"])].entity_type.isin(["Agent-Freehand"]).any() :
            d=d[~(d.chat_text.isna())]
            cust_index=d[d.entity_type=="Customer"].index.tolist()
            agent_index=d[d.entity_type.isin(["System","Agent-Freehand"])&(d.part_type.isin(["comment","open","close"]))].index.tolist()
            index_diff=[]
            for indx,agent in enumerate(agent_index):

                for _,cust in enumerate(cust_index):
                    if indx==0:
                        index_diff.append((agent,cust,cust-agent))
                        initial_agent_index=agent

                    else:
                        if cust>agent:
                            index_diff.append((agent,cust,cust-agent))
                            initial_agent_index=agent

            indx_diff=pd.DataFrame(index_diff,columns=["agent_indx","cust_indx","diff"])
                    
            
            def is_index_lesser(a,b,c):
                if a<b:
                    return True
                else:
                    return False
            indx_diff["is_lesser"]=indx_diff.apply(lambda x: is_index_lesser(a=x["agent_indx"],b=x["cust_indx"],c=x["diff"]),axis=1)
            sub=indx_diff[indx_diff.is_lesser==True]
            if len(sub)>0:
                sub=sub.groupby("agent_indx",as_index=False)["diff"].min().merge(sub,on=["agent_indx","diff"])
                sub=sub.groupby("cust_indx",as_index=False)["diff"].max().merge(sub,on=["cust_indx","diff"])
                da={}
                di=[]
                agntIndx=[]
                custIndx=[]
                for i in sub.cust_indx.unique():
                    di.append(sub[sub.cust_indx==i]["diff"].min())
                    agntIndx.append(sub[(sub["diff"]==sub[sub.cust_indx==i]["diff"].min())&(sub.cust_indx==i)].agent_indx.iloc[0])
                    custIndx.append(i)
                    da["diff"]=di
                    da["agent_indx"]=agntIndx
                    da["cust_indx"]=custIndx
                sub=pd.DataFrame(da)

                indx=[]
                for cust,agent in zip(sub.cust_indx,sub.agent_indx):
                    indx.append((cust,agent))
                time=[]
                for i in indx:
                    cust_time=pd.to_datetime(d.loc[i[0],'date_time'])
                    agnt_time=pd.to_datetime(d.loc[i[1],'date_time'])
                    time.append((cust_time-agnt_time).total_seconds())
                return np.median(time)/60
            else:
                return np.NaN
        else:
            time=[]
            if d[d.part_type.isin(["open","close","comment"])].entity_type.isin(["Agent-Freehand"]).any():
                agent_init_seq=d[(d.part_type.isin(["comment","open","close"]))&(d.entity_type=="Agent-Freehand")].sequence.iloc[0]
                agent_end_seq=d[(d.part_type.isin(["comment","open","close"]))&(d.entity_type=="Agent-Freehand")].sequence.iloc[-1]
                agent_init_time=pd.to_datetime(d[d.sequence==agent_init_seq]['date_time'].iloc[0])
                agent_end_time=pd.to_datetime(d[d.sequence==agent_end_seq]['date_time'].iloc[0])

                time.append((agent_end_time-agent_init_time).total_seconds())
                return np.median(time)
            else:
                return np.NaN
    else:
        return np.NaN
def max_agent_system_waiting_time(df):
        
    
    d=df[df.part_type.isin(["comment","open","close","assignment","note"])]
    d.reset_index(drop=True,inplace=True)
    if d.entity_type.isin(["Agent-Freehand","System"]).any():
        d=d[d.index>=d[d.entity_type.isin(["Agent-Freehand","System"])].index[0]]

        if d.entity_type.isin(["Customer"]).any() & d[d.part_type.isin(["open","close","comment"])].entity_type.isin(["Agent-Freehand"]).any() :
            d=d[~(d.chat_text.isna())]
            cust_index=d[d.entity_type=="Customer"].index.tolist()
            agent_index=d[d.entity_type.isin(["System","Agent-Freehand"])&(d.part_type.isin(["comment","open","close"]))].index.tolist()
            index_diff=[]
            for indx,agent in enumerate(agent_index):

                for _,cust in enumerate(cust_index):
                    if indx==0:
                        index_diff.append((agent,cust,cust-agent))
                        initial_agent_index=agent

                    else:
                        if cust>agent:
                            index_diff.append((agent,cust,cust-agent))
                            initial_agent_index=agent

            indx_diff=pd.DataFrame(index_diff,columns=["agent_indx","cust_indx","diff"])
                    
            
            def is_index_lesser(a,b,c):
                if a<b:
                    return True
                else:
                    return False
            indx_diff["is_lesser"]=indx_diff.apply(lambda x: is_index_lesser(a=x["agent_indx"],b=x["cust_indx"],c=x["diff"]),axis=1)
            sub=indx_diff[indx_diff.is_lesser==True]
            if len(sub)>0:
                sub=sub.groupby("agent_indx",as_index=False)["diff"].min().merge(sub,on=["agent_indx","diff"])
                sub=sub.groupby("cust_indx",as_index=False)["diff"].max().merge(sub,on=["cust_indx","diff"])
                da={}
                di=[]
                agntIndx=[]
                custIndx=[]
                for i in sub.cust_indx.unique():
                    di.append(sub[sub.cust_indx==i]["diff"].min())
                    agntIndx.append(sub[(sub["diff"]==sub[sub.cust_indx==i]["diff"].min())&(sub.cust_indx==i)].agent_indx.iloc[0])
                    custIndx.append(i)
                    da["diff"]=di
                    da["agent_indx"]=agntIndx
                    da["cust_indx"]=custIndx
                sub=pd.DataFrame(da)

                indx=[]
                for cust,agent in zip(sub.cust_indx,sub.agent_indx):
                    indx.append((cust,agent))
                time=[]
                for i in indx:
                    cust_time=pd.to_datetime(d.loc[i[0],'date_time'])
                    agnt_time=pd.to_datetime(d.loc[i[1],'date_time'])
                    time.append((cust_time-agnt_time).total_seconds())
                return max(time)/60
            else:
                return np.NaN
        else:
            time=[]
            if d[d.part_type.isin(["open","close","comment"])].entity_type.isin(["Agent-Freehand"]).any():
                agent_init_seq=d[(d.part_type.isin(["comment","open","close"]))&(d.entity_type=="Agent-Freehand")].sequence.iloc[0]
                agent_end_seq=d[(d.part_type.isin(["comment","open","close"]))&(d.entity_type=="Agent-Freehand")].sequence.iloc[-1]
                agent_init_time=pd.to_datetime(d[d.sequence==agent_init_seq]['date_time'].iloc[0])
                agent_end_time=pd.to_datetime(d[d.sequence==agent_end_seq]['date_time'].iloc[0])

                time.append((agent_end_time-agent_init_time).total_seconds())
                return max(time)
            else:
                return np.NaN
    else:
        return np.NaN
def min_agent_system_waiting_time(df):
        
    
    d=df[df.part_type.isin(["comment","open","close","assignment","note"])]
    d.reset_index(drop=True,inplace=True)
    if d.entity_type.isin(["Agent-Freehand","System"]).any():
        d=d[d.index>=d[d.entity_type.isin(["Agent-Freehand","System"])].index[0]]

        if d.entity_type.isin(["Customer"]).any() & d[d.part_type.isin(["open","close","comment"])].entity_type.isin(["Agent-Freehand"]).any() :
            d=d[~(d.chat_text.isna())]
            cust_index=d[d.entity_type=="Customer"].index.tolist()
            agent_index=d[d.entity_type.isin(["System","Agent-Freehand"])&(d.part_type.isin(["comment","open","close"]))].index.tolist()
            index_diff=[]
            for indx,agent in enumerate(agent_index):

                for _,cust in enumerate(cust_index):
                    if indx==0:
                        index_diff.append((agent,cust,cust-agent))
                        initial_agent_index=agent

                    else:
                        if cust>agent:
                            index_diff.append((agent,cust,cust-agent))
                            initial_agent_index=agent

            indx_diff=pd.DataFrame(index_diff,columns=["agent_indx","cust_indx","diff"])
                    
            
            def is_index_lesser(a,b,c):
                if a<b:
                    return True
                else:
                    return False
            indx_diff["is_lesser"]=indx_diff.apply(lambda x: is_index_lesser(a=x["agent_indx"],b=x["cust_indx"],c=x["diff"]),axis=1)
            sub=indx_diff[indx_diff.is_lesser==True]
            if len(sub)>0:
                sub=sub.groupby("agent_indx",as_index=False)["diff"].min().merge(sub,on=["agent_indx","diff"])
                sub=sub.groupby("cust_indx",as_index=False)["diff"].max().merge(sub,on=["cust_indx","diff"])
                da={}
                di=[]
                agntIndx=[]
                custIndx=[]
                for i in sub.cust_indx.unique():
                    di.append(sub[sub.cust_indx==i]["diff"].min())
                    agntIndx.append(sub[(sub["diff"]==sub[sub.cust_indx==i]["diff"].min())&(sub.cust_indx==i)].agent_indx.iloc[0])
                    custIndx.append(i)
                    da["diff"]=di
                    da["agent_indx"]=agntIndx
                    da["cust_indx"]=custIndx
                sub=pd.DataFrame(da)

                indx=[]
                for cust,agent in zip(sub.cust_indx,sub.agent_indx):
                    indx.append((cust,agent))
                time=[]
                for i in indx:
                    cust_time=pd.to_datetime(d.loc[i[0],'date_time'])
                    agnt_time=pd.to_datetime(d.loc[i[1],'date_time'])
                    time.append((cust_time-agnt_time).total_seconds())
                return min(time)/60
            else:
                return np.NaN
        else:
            time=[]
            if d[d.part_type.isin(["open","close","comment"])].entity_type.isin(["Agent-Freehand"]).any():
                agent_init_seq=d[(d.part_type.isin(["comment","open","close"]))&(d.entity_type=="Agent-Freehand")].sequence.iloc[0]
                agent_end_seq=d[(d.part_type.isin(["comment","open","close"]))&(d.entity_type=="Agent-Freehand")].sequence.iloc[-1]
                agent_init_time=pd.to_datetime(d[d.sequence==agent_init_seq]['date_time'].iloc[0])
                agent_end_time=pd.to_datetime(d[d.sequence==agent_end_seq]['date_time'].iloc[0])

                time.append((agent_end_time-agent_init_time).total_seconds())
                return min(time)
            else:
                return np.NaN
    else:
        return np.NaN
def duration(df):
    if df.part_type.isin(["conversation_rating_changed"]).any():
        conv_seq=df[df.part_type.isin(["conversation_rating_changed"])].sequence.iloc[-1]
        df=df[df.sequence<conv_seq]
    else:
        df=df
    if df.part_type.isin(["close"]).any():
        close_seq=df[df.part_type.isin(["close"])].sequence.tolist()
        time=[]
        if len(close_seq)<2:
            initial_seq=df.sequence.iloc[0]
            opening_time=pd.to_datetime(df[df.sequence==initial_seq].date_time.iloc[0])
            close_time=pd.to_datetime(df[df.sequence==close_seq[0]].date_time.iloc[0])
            time.append((close_time-opening_time).total_seconds()/60)
            return np.median(time)
        else:
            for indx,close in enumerate(close_seq):
                subdf=df[df.sequence<=close]
                if len(subdf)>1:
                    opening_time=pd.to_datetime(subdf.date_time.iloc[0])
                    close_time=pd.to_datetime(subdf[subdf.sequence==close].date_time.iloc[0])
                    time.append((close_time-opening_time).total_seconds()/60)
            return np.median(time)
    else:
        return np.NaN
def max_duration(df):
    if df.part_type.isin(["conversation_rating_changed"]).any():
        conv_seq=df[df.part_type.isin(["conversation_rating_changed"])].sequence.iloc[-1]
        df=df[df.sequence<conv_seq]
    else:
        df=df
    if df.part_type.isin(["close"]).any():
        close_seq=df[df.part_type.isin(["close"])].sequence.tolist()
        time=[]
        if len(close_seq)<2:
            initial_seq=df.sequence.iloc[0]
            opening_time=pd.to_datetime(df[df.sequence==initial_seq].date_time.iloc[0])
            close_time=pd.to_datetime(df[df.sequence==close_seq[0]].date_time.iloc[0])
            time.append((close_time-opening_time).total_seconds()/60)
            return max(time)
        else:
            for indx,close in enumerate(close_seq):
                subdf=df[df.sequence<=close]
                if len(subdf)>1:
                    opening_time=pd.to_datetime(subdf.date_time.iloc[0])
                    close_time=pd.to_datetime(subdf[subdf.sequence==close].date_time.iloc[0])
                    time.append((close_time-opening_time).total_seconds()/60)
            return max(time)
    else:
        return np.NaN
def min_duration(df):
    if df.part_type.isin(["conversation_rating_changed"]).any():
        conv_seq=df[df.part_type.isin(["conversation_rating_changed"])].sequence.iloc[-1]
        df=df[df.sequence<conv_seq]
    else:
        df=df
    if df.part_type.isin(["close"]).any():
        close_seq=df[df.part_type.isin(["close"])].sequence.tolist()
        time=[]
        if len(close_seq)<2:
            initial_seq=df.sequence.iloc[0]
            opening_time=pd.to_datetime(df[df.sequence==initial_seq].date_time.iloc[0])
            close_time=pd.to_datetime(df[df.sequence==close_seq[0]].date_time.iloc[0])
            time.append((close_time-opening_time).total_seconds()/60)
            return min(time)
        else:
            for indx,close in enumerate(close_seq):
                subdf=df[df.sequence<=close]
                if len(subdf)>1:
                    opening_time=pd.to_datetime(subdf.date_time.iloc[0])
                    close_time=pd.to_datetime(subdf[subdf.sequence==close].date_time.iloc[0])
                    time.append((close_time-opening_time).total_seconds()/60)
            return min(time)
    else:
        return np.NaN
def time_to_solution_delivered(df):
    
    if df.chat_text.str.contains("#solutiondelivered",case=False,na=False).any():
        solution_seq=df[df.chat_text.str.contains("#solutiondelivered",case=False,na=False)].sequence.tolist()
        time=[]
        for indx,seq in tqdm(enumerate(solution_seq)):
            if indx==0:
                if seq==1:
                    prev_seq=seq
                    pass
                else:
                    subdf=df[df.sequence<=seq]

                    if subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment","assignment"]))].shape[0]<1:
                        if subdf[subdf.part_type.isin(["open","close","comment","assignment"])].entity_type.isin(["System"]).any():
                            agent_first_cmnt_time=subdf[(subdf.entity_type.isin(["Agent-Freehand","System"]))&(subdf.part_type.isin(["open","close","comment","assignment"]))].date_time.iloc[0]
                            solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                            time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                            prev_seq=seq
                        elif subdf[subdf.part_type.isin(["open","close","comment","assignment"])].entity_type.isin(["Agent-Freehand"]).any():
                            
                            agent_first_cmnt_time=subdf[(subdf.entity_type.isin(["Agent-Freehand","System"]))&(subdf.part_type.isin(["open","close","comment","assignment"]))].date_time.iloc[0]
                            solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                            time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                            prev_seq=seq
                        else:
                            prev_seq=seq
                            pass
                            
                    else:
                        agent_first_cmnt_time=subdf[(subdf.entity_type.isin(["Agent-Freehand"]))&(subdf.part_type.isin(["open","close","comment","assignment"]))].date_time.iloc[0]
                        solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                        time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                        prev_seq=seq

            else:
                subdf=df[(df.sequence>prev_seq)&(df.sequence<=seq)]
                if len(subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment","note"]))])>0:
                    agent_first_cmnt_time=subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment","note"]))].date_time.iloc[0]
                    solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                    time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                    prev_seq=seq
                else:
                    prev_seq=seq
                    pass
                    
        return np.median(time)
    else:
        if df.entity_type.isin(["Customer"]).any() & (df[df.part_type.isin(["close","comment"])].entity_type.isin(["Agent-Freehand"]).any()):
            first_ping_time=df[df.sequence==1].date_time.iloc[0]
            cust_last_seq=df[df.entity_type=="Customer"].sequence.iloc[-1]
            last_ping_time=df[df.sequence==cust_last_seq].date_time.iloc[0]
            return (last_ping_time-first_ping_time).total_seconds()/60
        else:
            return np.NaN
def max_time_to_solution_delivered(df):
    
    if df.chat_text.str.contains("#solutiondelivered",case=False,na=False).any():
        solution_seq=df[df.chat_text.str.contains("#solutiondelivered",case=False,na=False)].sequence.tolist()
        time=[]
        for indx,seq in tqdm(enumerate(solution_seq)):
            if indx==0:
                if seq==1:
                    prev_seq=seq
                    pass
                else:
                    subdf=df[df.sequence<=seq]

                    if subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment","assignment"]))].shape[0]<1:
                        if subdf[subdf.part_type.isin(["open","close","comment","assignment"])].entity_type.isin(["System"]).any():
                            agent_first_cmnt_time=subdf[(subdf.entity_type.isin(["Agent-Freehand","System"]))&(subdf.part_type.isin(["open","close","comment","assignment"]))].date_time.iloc[0]
                            solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                            time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                            prev_seq=seq
                        elif subdf[subdf.part_type.isin(["open","close","comment","assignment"])].entity_type.isin(["Agent-Freehand"]).any():
                            
                            agent_first_cmnt_time=subdf[(subdf.entity_type.isin(["Agent-Freehand","System"]))&(subdf.part_type.isin(["open","close","comment","assignment"]))].date_time.iloc[0]
                            solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                            time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                            prev_seq=seq
                        else:
                            prev_seq=seq
                            pass
                            
                    else:
                        agent_first_cmnt_time=subdf[(subdf.entity_type.isin(["Agent-Freehand"]))&(subdf.part_type.isin(["open","close","comment","assignment"]))].date_time.iloc[0]
                        solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                        time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                        prev_seq=seq

            else:
                subdf=df[(df.sequence>prev_seq)&(df.sequence<=seq)]
                if len(subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment","note"]))])>0:
                    agent_first_cmnt_time=subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment","note"]))].date_time.iloc[0]
                    solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                    time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                    prev_seq=seq
                else:
                    prev_seq=seq
                    pass
                    
        if len(time)>0:
            return max(time)
    else:
        if df.entity_type.isin(["Customer"]).any() & (df[df.part_type.isin(["close","comment"])].entity_type.isin(["Agent-Freehand"]).any()):
            first_ping_time=df[df.sequence==1].date_time.iloc[0]
            cust_last_seq=df[df.entity_type=="Customer"].sequence.iloc[-1]
            last_ping_time=df[df.sequence==cust_last_seq].date_time.iloc[0]
            return (last_ping_time-first_ping_time).total_seconds()/60
        else:
            return np.NaN
def min_time_to_solution_delivered(df):
    
    if df.chat_text.str.contains("#solutiondelivered",case=False,na=False).any():
        solution_seq=df[df.chat_text.str.contains("#solutiondelivered",case=False,na=False)].sequence.tolist()
        time=[]
        for indx,seq in tqdm(enumerate(solution_seq)):
            if indx==0:
                if seq==1:
                    prev_seq=seq
                    pass
                else:
                    subdf=df[df.sequence<=seq]

                    if subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment","assignment"]))].shape[0]<1:
                        if subdf[subdf.part_type.isin(["open","close","comment","assignment"])].entity_type.isin(["System"]).any():
                            agent_first_cmnt_time=subdf[(subdf.entity_type.isin(["Agent-Freehand","System"]))&(subdf.part_type.isin(["open","close","comment","assignment"]))].date_time.iloc[0]
                            solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                            time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                            prev_seq=seq
                        elif subdf[subdf.part_type.isin(["open","close","comment","assignment"])].entity_type.isin(["Agent-Freehand"]).any():
                            
                            agent_first_cmnt_time=subdf[(subdf.entity_type.isin(["Agent-Freehand","System"]))&(subdf.part_type.isin(["open","close","comment","assignment"]))].date_time.iloc[0]
                            solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                            time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                            prev_seq=seq
                        else:
                            prev_seq=seq
                            pass
                            
                    else:
                        agent_first_cmnt_time=subdf[(subdf.entity_type.isin(["Agent-Freehand"]))&(subdf.part_type.isin(["open","close","comment","assignment"]))].date_time.iloc[0]
                        solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                        time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                        prev_seq=seq

            else:
                subdf=df[(df.sequence>prev_seq)&(df.sequence<=seq)]
                if len(subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment","note"]))])>0:
                    agent_first_cmnt_time=subdf[(subdf.entity_type=="Agent-Freehand")&(subdf.part_type.isin(["open","close","comment","note"]))].date_time.iloc[0]
                    solution_time=subdf[subdf.sequence==seq].date_time.iloc[0]
                    time.append((solution_time-agent_first_cmnt_time).total_seconds()/60)
                    prev_seq=seq
                else:
                    prev_seq=seq
                    pass
                    
        if len(time)>0:
            return min(time)
    else:
        if df.entity_type.isin(["Customer"]).any() & (df[df.part_type.isin(["close","comment"])].entity_type.isin(["Agent-Freehand"]).any()):
            first_ping_time=df[df.sequence==1].date_time.iloc[0]
            cust_last_seq=df[df.entity_type=="Customer"].sequence.iloc[-1]
            last_ping_time=df[df.sequence==cust_last_seq].date_time.iloc[0]
            return (last_ping_time-first_ping_time).total_seconds()/60
        else:
            return np.NaN




def cust_pings(df):
    pings=df[df.entity_type=="Customer"]
    return len(pings)
def agent_pings(df):
    pings=df[(df.entity_type=="Agent-Freehand")&(df.part_type.isin(['open','close','comment']))]
    return len(pings)
def number_sd_tag(df):
    tot=df.chat_text.str.contains("#solutiondelivered",na=False,case=False).sum()
    return tot
def number_of_agents(df):
    num_agents=df[~(df.entity_name=="System")&(df.part_type.isin(['open','close','comment']))&(df.entity_type=="Agent-Freehand")].entity_name.nunique()
    return num_agents
def created_date(df):
    return df.created_at.iloc[0]
def Ratings(df):
    return df["conversation_rating.rating"].iloc[-1]
def All_PA_Tags(df):
    tags=df["tags"].iloc[0]
    if type(tags)!=list:
        while type(tags)!=list:
            tags=ast.literal_eval(tags)
            print(tags)
            if len(tags)>0:
                for i in tags:
                    if str(i["name"]).__contains__("SG-PA"):
                        return i["name"]
                    
    else:
        if len(tags)>0:
            for i in tags:
                if str(i["name"]).__contains__("SG-PA"):
                    return i["name"]
                
def escalated(df):
    if any(df.chat_text.str.contains("#HelpNeededL1toL2",na=False,case=False)):
        return True
    else:
        return False
def time_to_assignment(df):
    return df['statistics.time_to_assignment'].iloc[0]
def time_to_admin_reply(df):
    return df['statistics.time_to_admin_reply'].iloc[0]
def time_to_first_close(df):
    return df['statistics.time_to_first_close'].iloc[0]
def time_to_last_close(df):
    return df['statistics.time_to_last_close'].iloc[0]
def median_time_to_reply(df):
    return df['statistics.median_time_to_reply'].iloc[0]
def count_reopens(df):
    return df['statistics.count_reopens'].iloc[0]
def count_assignments(df):
    return df['statistics.count_assignments'].iloc[0]
def conv_parts(df):
    return df['statistics.count_conversation_parts'].iloc[0]
def channel(df):
    return df.channel.iloc[0]
def is_controllable(df):
    controllable=[]
    for _,row in df.loc[:,["abandoned","self-resolved","is_duplicate",'spam']].iterrows() :
        if ((row[0]+row[1]+row[2]+row[3])>0):
            controllable.append(1)#Un-controllable
        else:
            controllable.append(0)
    return controllable
def agent_nps(df):
    ratings_agnet=df[df.entity_type=="Agent-Freehand"].groupby("entity_name",as_index=False)['conversation_rating.rating'].value_counts()
    def rating_avg(a,b):
        c=sum(a*b)/sum(b)
        return c
    def rating_avg_single(a):
        return a*.65
    nps=ratings_agnet.groupby("entity_name").apply(lambda x: rating_avg_single(a=x["conversation_rating.rating"].iloc[0]) if len(x)<2 else rating_avg(a=x["conversation_rating.rating"],b=x["count"])   )
    with open(pa_cluster_path+r'\agent_nps.pickle','rb') as f:
        agent_nps_map=pickle.load(f)
    agent_nps_map_new={}
    for i in range(len(nps)):
        agent_nps_map_new[nps.index[i]]=nps[i]
    final_nps={}
    for i in agent_nps_map.keys():
        if i in agent_nps_map_new.keys():
            prev_nps=agent_nps_map[i]
            new_nps=agent_nps_map_new[i]
            mean_nps=(.6*prev_nps)+(.4*new_nps)
            final_nps[i]=mean_nps
        else:
            prev_nps=agent_nps_map[i]
            final_nps[i]=prev_nps
    for i in agent_nps_map_new.keys():
        if i not in agent_nps_map.keys():
            new_nps=agent_nps_map_new[i]
            final_nps[i]=new_nps
            
            
    with open(agent_list_path+r'\agent_nps.pickle', 'wb') as handle:
        pickle.dump(final_nps, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
         
    return final_nps
# def ticket_nps(df,agent_nps_map):
#     with open(pa_cluster_path+r'\agent_nps.pickle','rb') as f:
#         ab=pickle.load(f)
    
    
    
#     nps=[]
#     agent_nps={}
#     agent_list=df[df.entity_type=="Agent-Freehand"].entity_name.unique()
#     if len(agent_list)>0:
#         for i in agent_list:
#             if i in ab.keys() and  i in agent_nps_map.keys():
#                 prev_agent_nps=ab[i]
#                 new_agent_nps=agent_nps_map[i]
#                 agent_nps[i]=(.8*prev_agent_nps)+(.2*new_agent_nps)
#             elif i in agent_nps_map.keys():
#                 agent_nps[i]=agent_nps_map[i]
#             elif i in ab.keys():
#                 agent_nps[i]=ab[i]
#             else:
#                 pass
            
#     if len(agent_list)>0:
#         for agent_name in agent_list:
#             try:
#                 nps.append(agent_nps[agent_name])
#             except:
#                 pass

#         return np.mean(nps)
#     else:
#         return 0
# def ticket_nps(df):
#     nps=[]
#     agent_list=df[df.entity_type=="Agent-Freehand"].entity_name.unique()
#     if len(agent_list)>0:
#         for agent_name in agent_list:
#             nps.append(df[df.entity_name==agent_name].agent_nps.iloc[0])

#         return np.mean(nps)
#     else:
#         return 0
def ticket_nps(df,agent_nps_map):
    nps=[]
    agent_list=df[df.entity_type=="Agent-Freehand"].entity_name.unique()
    if len(agent_list)>0:
        for agent_name in agent_list:
            if agent_name in agent_nps_map.keys():
                nps.append(agent_nps_map[agent_name])

        return np.mean(nps)
    else:
        return 0
    