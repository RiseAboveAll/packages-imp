import pandas as pd
import numpy as np
from data_prep_features import *
from feature_engineering import *
from path import *
from read_write_data import *
from categorical_feature import *
from text_features import *
from text_preprocess import *
from numeric_data_preprocess import *
def data_prep(df,is_train):
    if is_train:
        df=df[~(df["conversation_rating.rating"].isna())]
    PA_Tags=df.groupby("session_id",as_index=False).apply(All_PA_Tags).rename(columns={None:"PA_Tags"})
    Cust_waiting_time=df.groupby("session_id",as_index=False).apply(cust_waiting_time).rename(columns={None:"Avg_Cust_Waiting_Time"})
    agnt_waiting_time=df.groupby("session_id",as_index=False).apply(avg_agent_system_waiting_time).rename(columns={None:"Avg_Agent_Wating_Time"})
    Avg_Time_To_Reply_After_Assignment=df.groupby("session_id",as_index=False).apply(agent_reply_time_after_assignment).rename(columns={None:"Avg_Time_To_Reply_After_Assignment"})
    Avg_Time_To_Solution=df.groupby("session_id",as_index=False).apply(time_to_solution_delivered).rename(columns={None:"Avg_Time_To_Solution"})
    created_dt=df.groupby("session_id",as_index=False).apply(created_date).rename(columns={None:"created_date"})
    rating=df.groupby("session_id",as_index=False).apply(Ratings).rename(columns={None:"Rating"})
    max_cust_wati_time=df.groupby("session_id",as_index=False).apply(max_cust_waiting_time).rename(columns={None:"Max_Cust_Waiting_Time"})
    min_cust_wati_time=df.groupby("session_id",as_index=False).apply(min_cust_waiting_time).rename(columns={None:"Min_Cust_Waiting_Time"})
    Duration=df.groupby("session_id",as_index=False).apply(duration).rename(columns={None:"Duration"})
    max_reply_time_after_assignment=df.groupby("session_id",as_index=False).apply(max_agent_reply_time_after_assignment).rename(columns={None:"Max_Time_To_Reply_After_Assignment"})
    min_reply_time_after_assignment=df.groupby("session_id",as_index=False).apply(min_agent_reply_time_after_assignment).rename(columns={None:"Min_Time_To_Reply_After_Assignment"})

    max_agent_waiting_time=df.groupby("session_id",as_index=False).apply(max_agent_system_waiting_time).rename(columns={None:"Max_Agent_Wating_Time"})
    min_agent_waiting_time=df.groupby("session_id",as_index=False).apply(min_agent_system_waiting_time).rename(columns={None:"Min_Agent_Wating_Time"})

    max_ticket_duration=df.groupby("session_id",as_index=False).apply(max_duration).rename(columns={None:"Max_Duration"})
    min_ticket_duration=df.groupby("session_id",as_index=False).apply(min_duration).rename(columns={None:"Min_Duration"})

    max_sol_time=df.groupby("session_id",as_index=False).apply(max_time_to_solution_delivered).rename(columns={None:"Max_Solution_Time"})
    min_sol_time=df.groupby("session_id",as_index=False).apply(min_time_to_solution_delivered).rename(columns={None:"Min_Solution_Time"})
    

    out=df.groupby("session_id").apply(abandoned_ticket)
    final=Cust_waiting_time.merge(agnt_waiting_time,on="session_id").merge(Avg_Time_To_Reply_After_Assignment,on="session_id")
    final=final.merge(Avg_Time_To_Solution,on="session_id").merge(created_dt,on="session_id").merge(rating,on="session_id")
    final=final.merge(max_cust_wati_time,on="session_id").merge(min_cust_wati_time,on="session_id")
    final=final.merge(max_reply_time_after_assignment,on="session_id").merge(min_reply_time_after_assignment,on="session_id")
    final=final.merge(min_agent_waiting_time,on="session_id").merge(max_agent_waiting_time,on="session_id")
    final=final.merge(max_ticket_duration,on="session_id").merge(min_ticket_duration,on="session_id").merge(max_sol_time,on="session_id")
    final=final.merge(min_sol_time,on="session_id")
    final['statistics.time_to_assignment']=final.session_id.map(df.groupby("session_id").apply(time_to_assignment))
    final['statistics.time_to_admin_reply']=final.session_id.map(df.groupby("session_id").apply(time_to_admin_reply))
    final['statistics.time_to_first_close']=final.session_id.map(df.groupby("session_id").apply(time_to_first_close))
    final['statistics.time_to_last_close']=final.session_id.map(df.groupby("session_id").apply(time_to_last_close))
    final['statistics.median_time_to_reply']=final.session_id.map(df.groupby("session_id").apply(median_time_to_reply))
    final['statistics.count_reopens']=final.session_id.map(df.groupby("session_id").apply(count_reopens))
    final['statistics.count_assignments']=final.session_id.map(df.groupby("session_id").apply(count_assignments))
    final['statistics.count_conversation_parts']=final.session_id.map(df.groupby("session_id").apply(conv_parts))
    final=final.merge(PA_Tags,on="session_id").merge(Duration,on="session_id")
    final['channel']=final.session_id.map(df.groupby("session_id").apply(channel))
    final.channel.fillna("conversation",inplace=True)
    final["New_Rating"]=final.Rating.map(lambda x : None if pd.isna(x) else 0 if x>3 else 1)
    final["abandoned"]=final.session_id.map(out)
    out=df.groupby("session_id").apply(spam_ticket)
    final['spam']=final.session_id.map(out)
    out=df.groupby("session_id").apply(irt)
    final["irt"]=final.session_id.map(out)
    agent_list_path=r'C:\Users\birhiman\Documents\Zenoti-Files\csat-dsat\New - Approach\CSAT-DSAT-Refinement\utils'
    agent_list=load_data(agent_list_path,r'\sutherland-agents.pkl')
    agent_list=agent_list.agent_email.unique().tolist()
    out=df.groupby("session_id").apply(sutherland_agent,agent_list)
    final["sutherland_agent"]=final.session_id.map(out)
    out=df.groupby("session_id").apply(self_resolve)
    final["self-resolved"]=final.session_id.map(out)
    duplicate=df.groupby("session_id").apply(duplicate_ticket)
    final["is_duplicate"]=final.session_id.map(duplicate)
    if is_train:
        final=final[final.spam==False]
        final=final[final.is_duplicate==False]
        
        
    final["is_controllable"]=is_controllable(final)
    final['abandoned']=final['abandoned'].map(lambda x : 1 if x==True else 0)
    final['self-resolved']=final['self-resolved'].map(lambda x : 1 if x==True else 0)
    pa_tags_cluster=pa_cluster_path+r'\pa_tags_cluster.pkl'
    with open(pa_tags_cluster,"r+") as output:
        tags=output.readlines() 
    tags=ast.literal_eval(tags[0])
    PA_Encoding(final,tags)
    out = df.groupby("session_id").apply(is_positive)
    final["Cust_Happy_after_Solution"]=final.session_id.map(out)
    if is_train:
        with open(pa_cluster_path+r'\agent_nps.pickle','rb') as f:
            agent_nps_map=pickle.load(f)
        out=df.groupby("session_id").apply(ticket_nps,agent_nps_map)
        final['ticket_nps']=final.session_id.map(out)
    else:
        agent_nps_map=agent_nps(df)
        out=df.groupby("session_id").apply(ticket_nps,agent_nps_map)
        final["ticket_nps"]=final.session_id.map(out)
    
    return final

    
    
                                     
    