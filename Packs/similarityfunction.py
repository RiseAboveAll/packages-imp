from preprocessing import clean_text
import os
import numpy as np
import pickle
#import pickle5 as pickle
from numba import jit
import pandas as pd
from collections import defaultdict  # For word frequency
import re
from nltk.tokenize import sent_tokenize
import numpy as np
from numpy import float32 as REAL, sum as np_sum, multiply as np_mult
from gensim.models.phrases import Phraser

from tqdm import tqdm

# model_name=r'\CSAT_DSAT_'

model_name=r'\zenoti_v1'
path=r'C:\Users\birhiman\Documents\Zenoti-Files\csat-dsat\New - Approach\CSAT-DSAT-Refinement\utils'
import sys
sys.path.insert(1,path)
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

s2v_usif_model_dict_v1 = load_obj(path+model_name+'s2v.pkl')
bigram_phraser_model_v1 = Phraser.load(path+model_name+'bigram.pkl')



def compute_sentence_vector(sentence, model_name = None):
    if ((model_name is None) | (model_name=='zenoti_v1')):
        s2v_usif_model_dict = s2v_usif_model_dict_v1
        sentence = bigram_phraser_model_v1[clean_text(sentence)]
#     elif model_name=="test_model_new":
#         s2v_usif_model_dict = s2v_usif_model_dict_v2
#         sentence = bigram_phraser_model_v2[clean_text(sentence)]
#     elif model_name=="cox_cust":
#        s2v_usif_model_dict = s2v_usif_model_dict_v3
#        sentence = bigram_phraser_model_v3[clean_text(sentence)]
    if len(sentence)==0:
        return None
    word_indices_new = [s2v_usif_model_dict["word_index"][word] for word in sentence if word in s2v_usif_model_dict["word_index"]]
    mem_new = np_sum(np_mult(s2v_usif_model_dict["word_vectors"][word_indices_new], 
                             s2v_usif_model_dict["word_weights"][word_indices_new][:, None]), axis=0)
    if len(word_indices_new)==0:
        return None
    mem_new *= 1/len(word_indices_new)
    sentence_vector = mem_new - mem_new.dot(s2v_usif_model_dict["w_comp_new"].transpose()).dot(s2v_usif_model_dict["w_comp_new"])
    return sentence_vector


@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
   # print(u.shape)
   # print(v.shape)
    assert(u.shape[0]== v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta

 

def get_combinations(ping,length,exceptions=None):
    all_combinations=[]
    if exceptions is not None:
        for sent in sent_tokenize(str(ping)):
            if not bool(re.search(exceptions,sent,re.IGNORECASE)):
                sent_split= sent.split()
                if len(sent_split)>=length:
                    combinations=[" ".join(sent_split[i:i+length]) for i in range(len(sent_split)-length+1)]
                else:
                    combinations=[sent]
            else:
                combinations=[]
            all_combinations.extend(combinations)
    else:
        for sent in sent_tokenize(str(ping)):
            sent_split= sent.split()
            if len(sent_split)>=length:
                combinations=[" ".join(sent_split[i:i+length]) for i in range(len(sent_split)-length+1)]
            else:
                combinations=[sent]
            all_combinations.extend(combinations)
    return all_combinations


def get_similarity(baselines, sentences, threshold, fixed_length=None,exceptions=None,model_name=None,debug=False):
    baselines_length=[len(baseline.split()) for baseline in baselines]
    baselines_embeddings= [compute_sentence_vector(baseline,model_name) for baseline in baselines]
    if fixed_length is None:
        min_length=min(baselines_length)
        max_length=max(baselines_length)
        if min_length-max_length>3:
            lengths_to_search= np.linspace(min_length,max_length,3,dtype=int)
        elif min_length==max_length:
            lengths_to_search= [min_length]
        else:
            lengths_to_search= np.arange(min_length,max_length,dtype=int)
    elif fixed_length=='sentence':
        lengths_to_search=[None]        
    else:
        lengths_to_search=fixed_length
    max_sim_ping=[]
    for row in sentences:
        max_sim_sent=[0]
        for r in lengths_to_search:
            if r is not None:
                combination_sentences=get_combinations(row,r,exceptions)
            else:
                if exceptions is not None:
                    combination_sentences=[sent for sent in sent_tokenize(row) if not bool(re.search(exceptions,sent,re.IGNORECASE))] 
                else:
                    combination_sentences=sent_tokenize(row)

            query_embed=[compute_sentence_vector(sent,model_name) for sent in combination_sentences]
            max_sim_sent.append(np.max(([[cosine_similarity_numba(sent_b,baseline) for baseline in baselines_embeddings] for sent_b in query_embed if sent_b is not None]),initial=0))
        max_sim_ping.append(max(max_sim_sent))
    if debug:
        return max_sim_ping,np.array(max_sim_ping)>=threshold
    else:
        return np.array(max_sim_ping)>=threshold


def analyse_baselines(baselines, sentences, fixed_length=None,exceptions=None,model_name=None):
    baselines_length=[len(baseline.split()) for baseline in baselines]
    baselines_embeddings= [compute_sentence_vector(baseline,model_name) for baseline in baselines]
 #   print(baselines_embeddings)
    if fixed_length is None:
        min_length=min(baselines_length)
        max_length=max(baselines_length)
        if min_length-max_length>3:
            lengths_to_search= np.linspace(min_length,max_length,3,dtype=int)
        elif min_length==max_length:
            lengths_to_search= [min_length]
        else:
            lengths_to_search= np.arange(min_length,max_length,dtype=int)
    elif fixed_length=='sentence':
        lengths_to_search=[None]        
    else:
        lengths_to_search=fixed_length

    max_sim_ping=[]
    out_df=[]
    for row in sentences:
        baseline_dict_out={'sent':row}
        for base_idx, baseline_embed in enumerate(baselines_embeddings):
            max_sim=0
            max_sent=''
            for r in lengths_to_search:
                if r is not None:
                    combination_sentences=get_combinations(row,r,exceptions)
                else:
                    if exceptions is not None:
                        combination_sentences=[sent for sent in sent_tokenize(row) if not bool(re.search(exceptions,sent,re.IGNORECASE))] 
                    else:
                        combination_sentences=sent_tokenize(row)

                query_embed=[compute_sentence_vector(sent,model_name) for sent in combination_sentences]
                for query_idx, query_sent in enumerate(query_embed):
                    if query_sent is not None:
                    #    print(type(query_sent),type(baseline_embed))
                        score=cosine_similarity_numba(query_sent,baseline_embed)
                        if score > max_sim:
                            max_sim=score
                            max_sent= combination_sentences[query_idx]

            baseline_dict_out.update({baselines[base_idx]:max_sent,baselines[base_idx]+'_sim':max_sim })
        out_df.append(baseline_dict_out)
    out_df= pd.DataFrame(out_df)
    out_df['max_sim']=out_df.select_dtypes(include=np.number).max(axis=1)

    return out_df