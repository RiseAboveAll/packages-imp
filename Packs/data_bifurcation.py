import pandas as pd
import numpy as np

def bifercation(final):
    final_cont=final[final.is_controllable==0]
    final_uncont=final[final.is_controllable==1]
    controllable_conversation_sgs_data=final_cont[(final_cont.sutherland_agent==1)&(final_cont.channel=="conversation")]
    controllable_conversation_non_sgs_data=final_cont[(final_cont.sutherland_agent==0)&(final_cont.channel=="conversation")]
    controllable_email_data=final_cont[~(final_cont.channel=="conversation")]
    uncontrollable_conversation_sgs_data=final_uncont[(final_uncont.sutherland_agent==1)&(final_uncont.channel=="conversation")]
    uncontrollable_conversation_non_sgs_data=final_uncont[(final_uncont.sutherland_agent==0)&(final_uncont.channel=="conversation")]
    uncontrollable_email_data=final_uncont[~(final_uncont.channel=="conversation")]
    final_email_data=pd.concat([controllable_email_data,uncontrollable_email_data],axis=0)
    return [controllable_conversation_sgs_data,controllable_conversation_non_sgs_data,uncontrollable_conversation_sgs_data,uncontrollable_conversation_non_sgs_data,final_email_data]