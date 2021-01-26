from __future__ import division
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import PositionRank
from gensim.models import KeyedVectors
import evaluation
import process_data
import os
from os.path import isfile, join
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
import pickle

def ensure_dir(dirName):
    if not os.path.exists(dirName):
        print 'making dir: ' + dirName
        os.makedirs(dirName)

def load_obj(filePath):
    with open(filePath, 'rb') as f:
        return pickle.load(f)
      
def process(args):


    # initialize the evaluation metrics vectors
    P, R, F1 = [0] * args.topK, [0] * args.topK, [0] * args.topK
    Rprec = 0.0
    bpref = 0.0

    docs = 0

    #files = [f for f in os.listdir(args.input_data) if isfile(join(args.input_data, f))]
    files = []
    doc_list_file = open(args.doc_list)
    for l in doc_list_file:
        files.append(l.strip())
    doc_list_file.close()
    
    predDir = None
    summ_5 = None
    summ_10 = None
    summ_15 = None
    
    if args.output_dir:
        predDir = args.output_dir.rstrip('/') + '/preds/'
        ensure_dir(predDir)
        
        summ_5 = open(predDir+'summary_5', 'w')
        summ_10 = open(predDir+'summary_10', 'w')
        summ_15 = open(predDir+'summary_15', 'w')
     
        
    for filename in files:
        if files.index(filename)%1000 == 0:
            print filename
        # if doc has passed the criteria then we save its text and gold
        text = process_data.read_input_file(args.input_data + filename)
        gold = process_data.read_gold_file(args.input_gold + filename)

        if text and gold:
            
            gold_stemmed = []
            for keyphrase in gold:
                keyphrase = [porter_stemmer.stem(w) for w in keyphrase.lower().split()]
                gold_stemmed.append(' '.join(keyphrase))
            
            #gold_stemmed = process_data.load_stemmed_gold_phrases(gold)
            
            # count the document
            docs += 1
            embeddings = load_obj(args.emb_dir.rstrip('/') + '/' + filename + '_all.pkl')
            #system = PositionRank.PositionRank(text, args.window, args.phrase_type, args.emb_dim, args.emb_file)
            system = PositionRank.PositionRank(text, args.window, args.phrase_type, args.emb_dim, embeddings)

            system.get_doc_words()

            system.candidate_selection()

            system.candidate_scoring(theme_mode = args.theme_mode, update_scoring_method=False)

            currentP, currentR, currentF1 =\
                evaluation.PRF_range(system.get_best_k(args.topK), gold_stemmed, k=args.topK)

            Rprec += evaluation.Rprecision(system.get_best_k(args.topK), gold_stemmed,
                                           k=len(gold_stemmed))

            bpref += evaluation.Bpref(system.get_best_k(args.topK), gold_stemmed)
            
            P = map(sum, zip(P, currentP))
            R = map(sum, zip(R, currentR))
            F1 = map(sum, zip(F1, currentF1))
            
            if predDir:
                opFile = open(predDir+filename, 'w')
                preds = system.get_best_k_with_scores(50)
                for p in preds:
                    opFile.write(p[0].strip()+ '\t' + str(p[1]).strip() +'\n')
                opFile.close()
                summ_5.write(filename + '\t' + str(currentP[4]) + '\t' + str(currentR[4]) + '\t' + str(currentF1[4]) + '\n')
                summ_10.write(filename + '\t' + str(currentP[9]) + '\t' + str(currentR[9]) + '\t' + str(currentF1[9]) + '\n')
                summ_15.write(filename + '\t' + str(currentP[14]) + '\t' + str(currentR[14]) + '\t' + str(currentF1[14]) + '\n')
                
            #print 'docs', docs

    print 'docs', docs

    P = [p/docs for p in P]
    R = [r/docs for r in R]
    F1 = [f/docs for f in F1]

    # print 'Rprecision =', Rprec / docs
    # print 'Bpref', bpref / docs
    print 'Evaluation metrics:'.ljust(20, ' '), 'Precision @k'.ljust(20, ' '), 'Recall @k'.ljust(20, ' '), 'F1-score @k'

    opFile = None
    if args.output_dir:
        ensure_dir(args.output_dir)
        opFile = open(args.output_dir.rstrip('/') + '/summary_results', 'w')
        opFile.write('@k\tPr\tRe\tnewF1\tF1\n')
        
        summ_5.close()
        summ_10.close()
        summ_15.close()
        
    for i in range(0, args.topK):
        new_F1 = 0.0
        if (P[i] + R[i]) != 0:
            new_F1 = 2 * P[i] * R[i] / (P[i]+R[i])
        print ''.ljust(20, ' '), \
            'Pr@{}'.format(i + 1).ljust(6, ' '), '{0:.5f}'.format(P[i]).ljust(13, ' '), \
            'Re@{}'.format(i + 1).ljust(6, ' '), '{0:.5f}'.format(R[i]).ljust(13, ' '), \
            'F1@{}'.format(i + 1).ljust(6, ' '), '{0:.5f}'.format(F1[i])
        if opFile:
            #opFile.write('Pr@{}'.format(i + 1).ljust(6, ' ') + '{0:.5f}'.format(P[i]).ljust(13, ' ') + 'Re@{}'.format(i + 1).ljust(6, ' ') + '{0:.5f}'.format(R[i]).ljust(13, ' ') + 'F1@{}'.format(i + 1).ljust(6, ' ') + '{0:.5f}'.format(F1[i]) + '\n')
            opFile.write('@'+str(i+1)+'\t'+str(P[i])+'\t'+str(R[i])+'\t'+str(new_F1)+'\t'+str(F1[i]) + '\n')
    if opFile:
        opFile.close()

def main():
    parser = ArgumentParser("PositionRank",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')
    
    parser.add_argument('--doc_list', nargs='?', required=True,
                      help='list of documents: 1 name in each line')                      

    parser.add_argument('--input_data', nargs='?', required=True,
                      help='Directory with text documents')

    parser.add_argument('--input_gold', nargs='?', required=True,
                      help='Directory with documents containing the gold')
                      
    parser.add_argument('--topK', default=15, type=int,
                      help='Top k predicted')

    parser.add_argument('--window', default=10, type=int,
                      help='Window used to add edges in the graph')

    parser.add_argument('--phrase_type', default='n_grams',
                      help='If you want n-grams or noun phrases')
                      
    parser.add_argument('--emb_dim', default=768, type=int,
                      help='embedding dimentions')
                      
    parser.add_argument('--emb_dir', default=None,
                      help='Path to the word embedding directory containing pickled dictionaries')        
    
    parser.add_argument('--output_dir', default=None,
                      help='Directory to write summary of results')        
                      
    parser.add_argument('--theme_mode', default='adj_noun_title',
                      help='Directory to write summary of results')  

    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    sys.exit(main())