import os
import numpy as np
import pickle
from datetime import datetime

def ensure_dir(dirName):
    if not os.path.exists(dirName):
        print('making dir: ' + dirName)
        os.makedirs(dirName)

def load_obj(filePath):
    with open(filePath, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, filePath):
    with open(filePath, 'wb') as output:
        pickle.dump(obj, output, 2)

def combine_embeds(ip_dict):
    op_dict = {}
    
    op_dict['cls'] = ip_dict['embeddings'][0]
    op_dict['mean'] = ip_dict['embeddings'].mean(0)
    op_dict['token'] = ip_dict['tokens']
    op_dict['embeddings'] = ip_dict['embeddings']
    
    #print np.array(op_dict['mean']).shape
    #print ip_dict['embeddings'].shape
    words_before = ip_dict['words']
    embds = {}
    counts = {}
    
    i = 0
    while i < len(words_before):
        curr_word = words_before[i]
        i += 1
        curr_emb = np.array(ip_dict['embeddings'][i])
        curr_count = 1
        while i < len(words_before):
            if words_before[i].startswith('##'):
                curr_word += words_before[i][2:]
                i += 1
                
                curr_emb += np.array(ip_dict['embeddings'][i])
                curr_count += 1
            else:
                break
        
        curr_emb = curr_emb/curr_count
        
        if curr_word not in embds:
            counts[curr_word] = 1.0
            embds[curr_word] = curr_emb
        else:
            embds[curr_word] += curr_emb
            counts[curr_word] += 1.0
        
    for w in embds:
        embds[w] = embds[w]/counts[w]
    
    op_dict['combined_embeddings'] = embds
    return op_dict


def main():
    """
    Python 2.7 code
    generates a final dictionay containing word embeddings, CLS representations, and mean embeddings for each document name listed in overlap_test_bl.txt file in each dataset directory
    it combines splitted words if they are splitted into multiple tokens by a model
    averages word embeddings of a same work at muktiple locations
    file structure expected:
        - dataset_name
            - abstracts : directory containing abstracts
            - overlap_test_bl.txt : file containing a list of test documents, 1 document name per line
            - MODEL_MODE_emb_fulltext_title 
                - FILE_NAME_fulltext.pkl: file contains words, corresponding tokens, and embeddings for title as an input to the model
                - FILE_NAME_fulltext.pkl: file contains words, corresponding tokens, and embeddings for title+abstract as an input to the model
    Generates word embeddings as directory structure below:
        - dataset_name
            - MODEL_MODE_emb_combined_fulltext_title 
                - FILE_NAME_all.pkl: CLS representation of only title, mean word embedding of only title, CLS representation of title + abstract, mean word embedding of title + abstract, word embeddings for all unique words
                
    """
    model_mode = 'scibert' # 'bert'
    dsDir = '' # directory containing the datasets
    
    datasets = ['hulth', 'semeval','krapivin', 'nus']
    #datasets = ['acm'] 
    
    
    for ds in datasets:
        print(ds)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("date and time =", dt_string)
        ipDir = dsDir + ds + '/' + model_mode + '_emb_fulltext_title/'
        opDir = dsDir + ds + '/' + model_mode + '_emb_combined_fulltext_title/'
        ensure_dir(opDir)
         
        ipFile = open(dsDir + ds +'/overlap_test_bl.txt') 
        
        for l in ipFile:
            l = l.strip()
            opFilePath = opDir+l+ '_all.pkl'
            
            if os.path.exists(ipDir+l+ '_fulltext.pkl'): # and (not os.path.exists(opFilePath)) :
                #print(l)
                
                full_text_dict = combine_embeds(load_obj(ipDir+l+ '_fulltext.pkl'))
                title_dict = combine_embeds(load_obj(ipDir+l+ '_title.pkl'))
                
                op_dict = {}
    
                op_dict['cls_ttl'] = title_dict['cls']
                op_dict['mean_ttl'] = title_dict['mean']
                op_dict['cls_all'] = full_text_dict['cls']
                op_dict['mean_all'] = full_text_dict['mean']
                op_dict['embeddings'] = full_text_dict['combined_embeddings']
                
                save_obj(op_dict, opFilePath)
                    
        ipFile.close()
    
if __name__ == "__main__":
  main()