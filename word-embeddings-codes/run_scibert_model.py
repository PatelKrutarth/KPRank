import os
import codecs
import pickle
import re
from datetime import datetime
import torch
from transformers import BertTokenizer, BertModel

def ensure_dir(dirName):
    if not os.path.exists(dirName):
        print('making dir: ' + dirName)
        os.makedirs(dirName)

def getText(filePath):
    text = None
    title = None
    if os.path.exists(filePath):
        with codecs.open(filePath, "r", encoding='utf-8') as f:
            lines = f.readlines()
            lines[0] = lines[0].strip()
            if not (lines[0].endswith(".") or lines[0].endswith(".") or lines[0].endswith("!")):
                lines[0] = lines[0]+'.'
            text = ' '.join(lines)
            title = lines[0]
        f.close()
        
    return text, title
   
def load_obj(filePath):
    with open(filePath, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, filePath):
    with open(filePath, 'wb') as output:
        pickle.dump(obj, output, 2)

def main():
    """
    Python 3.7 code
    Download SciBERT (scibert_scivocab_uncased) model from: https://github.com/allenai/scibert
    generates wordembeddings for each document name listed in overlap_test_bl.txt file in each dataset directory
    file structure expected:
        - dataset_name
            - abstracts : directory containing abstracts
            - overlap_test_bl.txt : file containing a list of test documents, 1 document name per line
    Generates word embeddings as directory structure below:
        - dataset_name
            - MODEL_MODE_emb_fulltext_title 
                - FILE_NAME_fulltext.pkl: file contains words, corresponding tokens, and embeddings for title as an input to the model
                - FILE_NAME_fulltext.pkl: file contains words, corresponding tokens, and embeddings for title+abstract as an input to the model
    """
    
    model_mode = 'scibert' # 'bert'
    dsDir = '' # directory containing the datasets
    
    do_lower_case = True
    model = None
    tokenizer = None
    ####### SciBERT model #########
    if model_mode == 'scibert':
        model_version = 'XXX/models/scibert/scibert_scivocab_uncased' # please change the path to a downloaded Scibert Model
        model = BertModel.from_pretrained(model_version)
        tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
        
    elif model_mode == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
    
    
    def embed_text(text, model):
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states[0]
    
    def embed_tokens(tokens, model):
        input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1, only 1 sentense
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states[0] # num_tokens (or num_words+2) * 768 dimentioanal output
    
    #datasets = ['hulth', 'semeval']
    #datasets = ['krapivin', 'nus']
    #datasets = ['nus']
    datasets = ['acm']
    
    
    for ds in datasets:
        print(ds)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("date and time =", dt_string)
        ipDir = dsDir + ds + '/abstracts/'
        opDir = dsDir + ds + '/'+model_mode+'_emb_fulltext_title/'
        ensure_dir(opDir)
        
        # opening a file containing a list of test documents, 1 document name per line
        ipFile = open(dsDir + ds +'/overlap_test_bl.txt')
        
        for l in ipFile:
            l = l.strip()
            opFilePath_fulltext = opDir + l + '_fulltext.pkl'
            opFilePath_title = opDir + l + '_title.pkl'
            
            if not os.path.exists(opFilePath_fulltext):
                #print(l)
            
                fulltext, title = getText(ipDir+l)
                
                fulltext = re.sub('\s+', ' ', fulltext).strip() # remove extra spaces and new lines
                title = re.sub('\s+', ' ', title).strip() # remove extra spaces and new lines
                
                fulltext_words = tokenizer.tokenize(fulltext)
                title_words = tokenizer.tokenize(title)
                
                fulltext_en_tokens = tokenizer.convert_tokens_to_ids(['[CLS]'] + fulltext_words[:510] + ['[SEP]'])
                title_en_tokens = tokenizer.convert_tokens_to_ids(['[CLS]'] + title_words[:510] + ['[SEP]'])
                
                
                fulltext_em = embed_tokens(fulltext_en_tokens, model).detach().numpy()
                title_em = embed_tokens(title_en_tokens, model).detach().numpy()
                
                fulltext_dict = {}
                title_dict = {}
                
                fulltext_dict['words'] = fulltext_words[:510]
                fulltext_dict['tokens'] = fulltext_en_tokens
                fulltext_dict['embeddings'] = fulltext_em
                
                title_dict['words'] = title_words[:510]
                title_dict['tokens'] = title_en_tokens
                title_dict['embeddings'] = title_em
                
                
                save_obj(fulltext_dict, opFilePath_fulltext)
                save_obj(title_dict, opFilePath_title)
                
        ipFile.close()
        

if __name__ == "__main__":
  main()