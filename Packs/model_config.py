from path import *


############### Controllable Conversation SGS ###############
ccs_params={}

ccs_params["subsample"]=.9

ccs_params["reg_alpha"]=1.50
ccs_params["gamma"]=.75
ccs_params["base_score"]=.3
ccs_params["n_estimators"]=100
ccs_params["random_state"]=0
ccs_params["colsample_bytree"]=.8
ccs_params["reg_lambda"]=50
grid_ccs_params={'max_depth':[3,4,5,6,7],'scale_pos_weight':[.01,.05,.09,1,10],'learning_rate':[.01,.09,.1,.2,.3],'n_estimators':[20,30,50,70,100],'subsample':[.6,.7,.8,.9],'gamma':[.5,.75,1,1.25],'reg_alpha':[.75,10,20,40,50],"reg_lambda":[10,30,40,50]}

ccs_config={'Imp_Cols':['x0_Appointment','x0_Misc','x0_Payment',"Cust_Happy_after_Solution","ticket_nps","Avg_Cust_Waiting_Time"],
            "base_model_name":base_model+r'\cont-conv-sgs-base.pkl'}


############### Un-Controllable Conversation SGS ###############
unccs_params={}
unccs_params['random_state']=0
unccs_params['max_depth']=3
unccs_params['n_estimators']=100
unccs_params['subsample']=.9
unccs_params['colsample_bytree']=.9
unccs_params['gamma']=.75
unccs_params['reg_alpha']=.25

grid_unccs_params={'max_depth':[3,5,7],'scale_pos_weight':[.1,.5,1,10],'learning_rate':[.01,.09,.1,.2,.3],'n_estimators':[20,30,50,70,100],"colsample_bytree":[.5,.6,.7,.8,.9],'subsample':[.6,.7,.8,.9],'gamma':[.5,.75,1,1.25],'reg_alpha':[.75,10,20,40,50],"reg_lambda":[10,30,40,50]}

unccs_config={'Imp_Cols':[  'x0_Payment',"Cust_Happy_after_Solution","ticket_nps","Avg_Cust_Waiting_Time","Max_Solution_Time","Avg_Time_To_Solution"],"base_model_name":base_model+r'\unccont-conv-sgs-base.pkl'}


############### Controllable Conversation NoN-SGS ###############
ccns_params={}
ccns_params['random_state']=0
ccns_params['max_depth']=3
ccns_params['n_estimators']=50
ccns_params['subsample']=.9
ccns_params['eta']=.09
ccns_params['gamma']=.75
ccns_params['reg_alpha']=.10
ccns_params['reg_lambda']=50

grid_ccns_params={'max_depth':[3,5,7],'n_estimators':[15,20,30],'subsample':[.7,.8,.9],'gamma':[.75,1,1.25],'reg_alpha':[.10,.75,10,20,40,50],"reg_lambda":[10,30,40,50,60,70]}

ccns_config={'Imp_Cols':[  'x0_Payment',"Cust_Happy_after_Solution","ticket_nps","Avg_Cust_Waiting_Time"],"base_model_name":base_model+r'\cont-conv-nonsgs-base.pkl'}


############### Un-Controllable Conversation NoN-SGS ###############
unccns_params={"random_state":0,"max_depth":3,"n_estimators":25,"colsample_bytree":.8,"subsample":.785,"reg_alpha":1.9,"gamma":.75,"eta":.091,"reg_lambda":10}

grid_unccns_params={'max_depth':[3,5,7],'n_estimators':[15,20,30],'subsample':[.7,.8,.9],'colsample_bytree':[.7,.8,.9],'gamma':[.75,1,1.25],'reg_alpha':[.10,.75,10,20,40,50],'eta':[.3,.2,.1,.091,.05,.01]}

unccns_config={"Imp_Cols":[  'x0_Payment',"Cust_Happy_after_Solution","ticket_nps","Avg_Cust_Waiting_Time","Avg_Agent_Wating_Time","irt"],
               "base_model_name":base_model+r'\uncont-conv-nsgs-base.pkl'}

############### Email ###############
email_params={"random_state":0,"max_depth":3,"n_estimators":30,"colsample_bytree":.8,"subsample":.785}

grid_email_params={'max_depth':[3,5,7],'n_estimators':[15,20,30,50,100],'subsample':[.7,.8,.9],'colsample_bytree':[.7,.8,.9],'gamma':[.75,1,1.25],"reg_lambda":[10,30,40,50,60,70]}

email_config={"Imp_Cols":[  'x0_Payment',"Cust_Happy_after_Solution","ticket_nps"],
               "base_model_name":base_model+r'\email-base.pkl'}


              


