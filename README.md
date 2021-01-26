# KPRank

Exploiting Position and Contextual Word Embeddings for Keyphrase Extraction from Scientific Papers

## Authors
[Krutarth Patel](https://patelkrutarth.github.io/) and [Cornelia Caragea](https://www.cs.uic.edu/~cornelia/)
### Usage

Contains two directory:
1. KPRank-codes : contains codes for KPRank
2. word-embeddings-codes : contains codes for generating word embeddings from SciBERT/BERT model

Required Python modules for the codes in both directories are in the requirments.txt file in each directory.

First, get word embeddings using codes from word-embeddings-codes:
```
$python3.7 run_scibert_model.py  (Python3.7)
$python prepare_bert_scibert_final_dicts.py (Python2.7)
```
Run KPRank from KPRank-codes:
```
$python __main__.py --doc_list DATASET/overlap_test_bl.txt --input_data DATASET/abstracts/ --input_gold DATASET/gold_ctr_unctr/ --emb_dim 768 --emb_dir DATASET/scibert_emb_combined_fulltext_title/ --output_dir DATASET/results/kprank --theme_mode adj_noun_title (Python2.7)
```
-\-doc\_list : list of documents with 1 name in each line
-\-input\_data : directory containing text documents
-\-input\_gold: directory containing gold-standard keyphrases for each document for evaluation purpose
The full list of command line options is available with $python \_\_main\_\_.py -\-help

### Citing
If you find PositionRank useful in your research, we ask that you cite the following paper:

> Krutarth Patel, and Cornelia Caragea. Exploiting Position and Contextual Word Embeddings for Keyphrase Extraction from Scientific Papers. Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics , 2021. 

